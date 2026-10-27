"""Provided interview utilities: data, model, training, sampling, and visualization.

Batch construction, alignment, and geometry losses live in the notebook. Training,
evaluation, and visualization accept those implementations explicitly; importing
this module does not load data, train a model, or write artifacts.
"""

import random
import re
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, NotRequired, Protocol, Self, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from torch import Tensor, nn

_MAX_ATOMS: Final[int] = 9  # Fixed distance-feature width inside the posterior only.
MAX_TOKENS: Final[int] = 128

type MetricName = Literal["posterior_mse", "prior_mse", "distance_mse", "kl"]


class SmilesTokens(NamedTuple):
    tokens: list[str]
    atom_token_indices: list[int]


class ConformerExample(TypedDict):
    id: str
    smiles: str
    coordinates: Float[np.ndarray, "n 3"]


class ConformerItem(TypedDict):
    token_ids: Int[Tensor, "t"]
    atom_token_indices: Int[Tensor, "n"]
    coordinates: NotRequired[Float[Tensor, "n 3"]]


class Metrics(TypedDict):
    posterior_mse: float
    prior_mse: float
    distance_mse: float
    kl: float


class HistoryRow(Metrics):
    epoch: int
    train_loss: float
    beta: float


class Batch(Protocol):
    """The fields the provided model consumes from the notebook's batch class."""

    token_ids: Int[Tensor, "b t"]
    token_mask: Bool[Tensor, "b t"]
    atom_token_indices: Int[Tensor, "b n"]
    atom_mask: Bool[Tensor, "b n"]
    coords: Float[Tensor, "b n 3"]

    def to(self, device: torch.device | str) -> Self: ...


type BatchFactory = Callable[[Sequence[ConformerItem]], Batch]
type AlignFn = Callable[[Tensor, Tensor, Tensor], Tensor]
type LossFn = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


def load_qm9(path: Path = Path(__file__).parent / "data" / "qm9_256.sdf") -> list[Mol]:
    """Read the bundled, prepared heavy-atom molecules."""
    if not path.is_file():
        raise FileNotFoundError(f"Include the prepared dataset with the notebook: {path}")
    mols: list[Mol] = []
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is None:
            raise ValueError("Invalid molecule in the prepared dataset")
        mols.append(mol)
    return mols


# Character IDs are fixed; H is allowed in bracket metadata, never as an output atom.
VOCAB: Final[dict[str, int]] = {
    token: i for i, token in enumerate(["<PAD>", *"HBCNOSPFIbcnopsrl0123456789[]()@-+=#$:/\\%."])
}
# RDKit validates chemistry. Match each whole bracket atom so its H count is not an atom.
ATOM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[(?:\d+)?(?P<bracket>[A-Z][a-z]?|[bcnops])[^]]*\]"
    r"|(?P<bare>Cl|Br|[BCNOSPFIbcnops])"
)


def tokenize(smi: str) -> SmilesTokens:
    """Character-tokenize RDKit-validated heavy SMILES and locate atom element starts."""
    atom_token_indices = [
        match.start("bracket") if match["bracket"] else match.start("bare")
        for match in ATOM_PATTERN.finditer(smi)
    ]

    return SmilesTokens(list(smi), atom_token_indices)


def embed_mol(mol: Mol) -> ConformerItem:
    """Encode a molecule, including coordinates when a conformer is present."""
    smi = Chem.MolToSmiles(mol, canonical=False)
    tokens, atom_token_indices = tokenize(smi)

    if len(tokens) > MAX_TOKENS:
        raise ValueError("Molecule exceeds model token limit")
    if unknown := set(tokens) - VOCAB.keys():
        raise ValueError(f"Unsupported SMILES tokens: {sorted(unknown)}")
    item: ConformerItem = {
        "token_ids": torch.tensor([VOCAB[token] for token in tokens], dtype=torch.long),
        "atom_token_indices": torch.tensor(atom_token_indices, dtype=torch.long),
    }
    if mol.GetNumConformers():
        item["coordinates"] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)

    return item


def describe_molecules(mols: Sequence[Mol]) -> list[ConformerExample]:
    """Keep display metadata separate from the lazily encoded dataset items."""
    return [
        {
            "id": mol.GetProp("_Name"),
            "smiles": Chem.MolToSmiles(mol, canonical=False),
            "coordinates": np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32),
        }
        for mol in mols
    ]


def random_rotations(
    b: int, device: torch.device, generator: torch.Generator | None = None
) -> Float[Tensor, "b 3 3"]:
    # A normalized isotropic Gaussian quaternion is uniform on SO(3).
    q = torch.randn(b, 4, device=device, generator=generator)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(b, 3, 3)


def _weighted_mean(
    x: Float[Tensor, "..."],
    weights: Bool[Tensor, "..."] | Float[Tensor, "..."],
    dim: int | tuple[int, ...] = -1,
) -> Float[Tensor, "..."]:
    """Average finite inputs over dim with nonnegative weights; zero totals return zero.

    Weights may omit trailing feature axes, e.g. b n weights for b n d values.
    """
    while weights.ndim < x.ndim:
        weights = weights.unsqueeze(-1)

    weights = weights.expand_as(x)
    denominator = weights.sum(dim=dim)
    denominator = denominator.masked_fill(denominator == 0, 1)

    return (x * weights).sum(dim=dim) / denominator


class SmilesConformerVAE(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, latent_dim: int = 8) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_TOKENS, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=2 * embed_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(embed_dim + _MAX_ATOMS, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(embed_dim, 2 * latent_dim), nn.Unflatten(-1, (latent_dim, 2))
        )
        self.coordinate_head = nn.Sequential(
            nn.Linear(2 * embed_dim + latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def encode(self, batch: Batch) -> Float[Tensor, "b n d"]:
        tokens = batch.token_ids
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        embeds = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        H = self.encoder(embeds, src_key_padding_mask=~batch.token_mask)
        indices = batch.atom_token_indices.unsqueeze(-1).expand(-1, -1, H.shape[-1])

        return H.gather(1, indices) * batch.atom_mask.unsqueeze(-1)

    def posterior(
        self,
        H: Float[Tensor, "b n d"],
        R: Float[Tensor, "b n d_r"],
        mask: Bool[Tensor, "b n"],
    ) -> tuple[Float[Tensor, "b d_z"], Float[Tensor, "b d_z"]]:
        n = R.shape[1]
        if (pad := _MAX_ATOMS - n) < 0:
            raise ValueError(f"Posterior supports at most {_MAX_ATOMS} atoms; got {n}")

        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (b, n, n)
        pairwise_dists = torch.cdist(R, R) * pair_mask  # (b, n, n)
        pairwise_dists = F.pad(pairwise_dists, [0, pad])  # (b, n, N_max)
        G = self.geometry_encoder(torch.cat([H, pairwise_dists / 3.0], dim=-1))  # (b, n, d_r)
        mu, logvar = self.posterior_head(_weighted_mean(G, mask, dim=1)).unbind(-1)

        return mu, logvar.clamp(-8, 8)

    def decode(
        self,
        H: Float[Tensor, "b n d"],
        Z: Float[Tensor, "b d_z"],
        mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, "b n d_r"]:
        H_global = _weighted_mean(H, mask, dim=1).unsqueeze(1).expand_as(H)
        latent = Z.unsqueeze(1).expand(-1, H.shape[1], -1)
        R = self.coordinate_head(torch.cat([H, H_global, latent], dim=-1))  # (b, n, d_r)

        return cast(Tensor, R * mask.unsqueeze(-1))

    def forward(
        self, batch: Batch, target: Float[Tensor, "b n d_r"]
    ) -> tuple[Float[Tensor, "b n 3"], Float[Tensor, "b l"], Float[Tensor, "b l"]]:
        mask = batch.atom_mask
        H = self.encode(batch)
        mu, logvar = self.posterior(H, target, mask)
        Z = mu + (logvar / 2).exp() * torch.randn_like(mu)

        return self.decode(H, Z, mask), mu, logvar


@torch.no_grad()
def evaluate(
    model: SmilesConformerVAE,
    loader: Iterable[Batch],
    *,
    loss_fn: LossFn,
    seed: int = 7,
) -> Metrics:
    """Evaluate supplied batches with posterior means and repeatable prior latents.

    Use a stable loader order to keep the prior latent for each molecule fixed.
    Metrics are averaged over the molecules actually yielded by the loader.
    """
    model.eval()
    device = next(model.parameters()).device
    totals = Metrics(posterior_mse=0.0, prior_mse=0.0, distance_mse=0.0, kl=0.0)
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    count = 0
    for raw_batch in loader:
        batch = raw_batch.to(device)
        mask, target = batch.atom_mask, batch.coords
        h = model.encode(batch)
        mu, logvar = model.posterior(h, target, mask)
        prediction = model.decode(h, mu, mask)
        rec, dist = loss_fn(prediction, target, mask)
        z = torch.randn(len(target), model.latent_dim, generator=generator, device=device)
        prior_rec, _ = loss_fn(model.decode(h, z, mask), target, mask)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
        values: list[tuple[MetricName, Float[Tensor, " b"]]] = [
            ("posterior_mse", rec),
            ("prior_mse", prior_rec),
            ("distance_mse", dist),
            ("kl", kl),
        ]
        for name, value in values:
            totals[name] += value.sum().item()
        count += len(target)
    if count == 0:
        raise ValueError("Evaluation loader yielded no molecules")
    return Metrics(
        posterior_mse=totals["posterior_mse"] / count,
        prior_mse=totals["prior_mse"] / count,
        distance_mse=totals["distance_mse"] / count,
        kl=totals["kl"] / count,
    )


def train(
    model: SmilesConformerVAE,
    loader: Iterable[Batch],
    eval_loader: Iterable[Batch],
    *,
    loss_fn: LossFn,
    epochs: int = 180,
    augment_rotations: bool = True,
    seed: int = 7,
) -> list[HistoryRow]:
    """Train with the notebook's loss callback and return epoch metrics for plotting.

    Both loaders must support repeated iteration (for example, DataLoaders or lists).
    The model's device is used for all batches; seed fixes evaluation prior samples.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history: list[HistoryRow] = [
        HistoryRow(
            epoch=0,
            train_loss=float("nan"),
            beta=0.0,
            **evaluate(model, eval_loader, loss_fn=loss_fn, seed=seed),
        )
    ]
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        beta = 0.1 * min(epoch / 30, 1.0)
        for raw_batch in loader:
            batch = raw_batch.to(device)
            target, mask = batch.coords, batch.atom_mask
            if augment_rotations:
                target = target @ random_rotations(len(target), device)
            prediction, mu, logvar = model(batch, target)
            rec, dist = loss_fn(prediction, target, mask)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)
            loss = (rec + 0.1 * dist + beta * kl).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)
        if count == 0:
            raise ValueError("Training loader yielded no molecules")
        metrics = evaluate(model, eval_loader, loss_fn=loss_fn, seed=seed)
        history.append(HistoryRow(epoch=epoch, train_loss=total / count, beta=beta, **metrics))
        if epoch == 1 or epoch % 30 == 0:
            print(
                f"Epoch {epoch:3d} | objective {total / count:.4f} | "
                f"posterior MSE {metrics['posterior_mse']:.4f} | "
                f"prior MSE {metrics['prior_mse']:.4f}"
            )
    print(f"Training time: {time.perf_counter() - start:.1f} seconds")
    return history


@torch.no_grad()
def sample_conformers(
    model: SmilesConformerVAE,
    batch_factory: BatchFactory,
    smiles: str,
    n_samples: int = 4,
    seed: int = 42,
    random_orientation: bool = True,
) -> Mol:
    """Sample from prepared heavy-atom SMILES, preserving their atom order."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    device = next(model.parameters()).device
    mol = Chem.MolFromSmiles(smiles)
    item = embed_mol(mol)
    batch = batch_factory([item] * n_samples).to(device)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    Z = torch.randn(n_samples, model.latent_dim, generator=generator, device=device)
    X = model.decode(model.encode(batch), Z, batch.atom_mask)
    if random_orientation:
        X = X @ random_rotations(n_samples, device, generator)
    coordinates_array = X[:, : mol.GetNumAtoms()].cpu().numpy()
    mol.RemoveAllConformers()
    for coordinates in coordinates_array:
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.Set3D(True)
        conformer.SetPositions(coordinates.astype(np.float64))
        mol.AddConformer(conformer, assignId=True)
    return mol


def add_display_hydrogens(mol: Mol, xyz: Float[np.ndarray, "n 3"]) -> Mol:
    """Place and UFF-relax hydrogens on a display copy with heavy atoms fixed."""
    display = Chem.Mol(mol)
    display.RemoveAllConformers()
    conformer = Chem.Conformer(display.GetNumAtoms())
    conformer.Set3D(True)
    conformer.SetPositions(np.asarray(xyz, dtype=np.float64))
    display.AddConformer(conformer, assignId=True)
    display = Chem.AddHs(display, addCoords=True)
    force_field = UFFGetMoleculeForceField(display)
    for atom in display.GetAtoms():
        if atom.GetAtomicNum() != 1:
            force_field.AddFixedPoint(atom.GetIdx())
    force_field.Initialize()
    if force_field.Minimize(maxIts=2000) != 0:
        warnings.warn("Display hydrogen relaxation did not converge.", stacklevel=2)
    return display


def draw_molecule(
    ax: Axes3D,
    mol: Mol,
    xyz: Float[np.ndarray, "n 3"],
    title: str,
    limits: Sequence[tuple[float, float]],
    *,
    as_reference: bool = False,
) -> None:
    """Draw a molecule, optionally as a faint background reference for an overlay."""
    alpha = 0.3 if as_reference else 1.0
    colors = {"H": "#d0d5dd", "C": "#344054", "N": "#4277e9", "O": "#e85858", "F": "#4dae66"}
    for bond in mol.GetBonds():
        points = xyz[[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        ax.plot(
            *points.T,
            color="#98a2b3" if as_reference else "#7d8799",
            lw=2 if as_reference else 1.5,
            alpha=alpha,
            zorder=1 if as_reference else 3,
        )
    for atom, point in zip(mol.GetAtoms(), xyz):
        symbol = atom.GetSymbol()
        ax.scatter(
            *point,
            color="#98a2b3" if as_reference else colors[symbol],
            alpha=alpha,
            depthshade=False,
            zorder=2 if as_reference else 4,
            s=35 if symbol == "H" else 90,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], title=title)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def plot_losses(
    history: Sequence[HistoryRow],
    n_molecules: int = 256,
    out_dir: Path = Path("artifacts"),
) -> None:
    """Show and save the reconstruction, training-objective, and KL curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    first, last = history[0], history[-1]
    reconstruction_metrics: tuple[tuple[MetricName, str], ...] = (
        ("posterior_mse", "Posterior reconstruction"),
        ("prior_mse", "Fixed-prior reconstruction"),
    )
    for key, label in reconstruction_metrics:
        print(
            f"{label}: {first[key]:.4f} → {last[key]:.4f} Å² "
            f"({100 * (1 - last[key] / first[key]):.1f}% decrease)"
        )
    print(f"Final fixed-prior aggregate RMSD: {np.sqrt(last['prior_mse']):.3f} Å")
    print(f"Final KL: {last['kl']:.6f} nats/molecule")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs[1:], [row["train_loss"] for row in history[1:]], color="tab:green")
    axes[0].set(title="Weighted training objective", ylabel="Training loss")
    axes[1].plot(
        epochs, [row["posterior_mse"] for row in history], label="Posterior mean", alpha=0.7
    )
    axes[1].plot(
        epochs, [row["prior_mse"] for row in history], label="Fixed prior samples", alpha=0.7
    )
    axes[1].set(title="Heavy-atom Kabsch reconstruction", ylabel="Mean squared distance (Å²)")
    axes[1].legend()
    axes[2].plot(epochs, [row["kl"] for row in history], color="tab:purple")
    axes[2].set(title="Posterior KL to $N(0, I)$", ylabel="KL (nats / molecule)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{n_molecules} QM9 training molecules · heavy atoms · no split", y=1.03)
    fig.tight_layout()
    plt.show()


def plot_conformers(
    model: SmilesConformerVAE,
    mols: Sequence[Mol],
    batch_factory: BatchFactory,
    align_fn: AlignFn,
    loss_fn: LossFn,
    *,
    seed: int = 7,
    n_molecules: int = 5,
    n_samples: int = 3,
    out_dir: Path = Path("artifacts"),
) -> None:
    """Sample, align, display, and export using the notebook's supplied implementations."""
    selected = random.Random(seed).sample(list(mols), k=min(n_molecules, len(mols)))
    selected_examples = describe_molecules(selected)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 3.2 * len(selected_examples)), layout="constrained")
    row_figures = fig.subfigures(nrows=len(selected_examples), ncols=1, squeeze=False)
    with Chem.SDWriter(str(out_dir / "generated_conformers.sdf")) as writer:
        for row, example in enumerate(selected_examples):
            generated = sample_conformers(
                model, batch_factory, example["smiles"], n_samples=n_samples, seed=42 + row
            )
            coords = torch.tensor(
                np.stack([c.GetPositions() for c in generated.GetConformers()]), dtype=torch.float32
            )
            reference = torch.tensor(example["coordinates"])[None].expand_as(coords)
            mask = torch.ones(coords.shape[:2], dtype=torch.bool)
            aligned = align_fn(coords, reference, mask).numpy()
            rmsds = np.sqrt(((aligned - example["coordinates"]) ** 2).sum(-1).mean(-1))
            print(example["id"], "|", example["smiles"], "| prior RMSDs (Å):", np.round(rmsds, 3))
            first_sample = coords[:1].expand_as(coords)
            diversity, _ = loss_fn(coords, first_sample, mask)
            print(
                "RMSD to first sample after alignment (Å):", np.round(diversity.sqrt().numpy(), 4)
            )
            generated.SetProp("reference_id", example["id"])
            for conformer in generated.GetConformers():
                writer.write(generated, confId=conformer.GetId())

            display_molecules = [
                add_display_hydrogens(generated, xyz) for xyz in [example["coordinates"], *aligned]
            ]
            geometries = [mol.GetConformer().GetPositions() for mol in display_molecules]
            stacked = np.concatenate(geometries)
            midpoint = (stacked.min(0) + stacked.max(0)) / 2
            radius = np.ptp(stacked, axis=0).max() / 2 + 0.4
            limits = list(zip(midpoint - radius, midpoint + radius))
            row_figure = row_figures[row, 0]
            row_figure.suptitle(f"{example['id']} · {example['smiles']}", fontsize=13)
            for i, xyz in enumerate(geometries[1:], start=1):
                # Explicit layers keep generated atoms and bonds above the faint reference.
                molecule_ax = cast(
                    Axes3D,
                    row_figure.add_subplot(1, n_samples, i, projection="3d", computed_zorder=False),
                )
                title = f"Prior sample {i} · RMSD {rmsds[i - 1]:.3f} Å"
                draw_molecule(
                    molecule_ax,
                    display_molecules[0],
                    geometries[0],
                    title,
                    limits,
                    as_reference=True,
                )
                draw_molecule(molecule_ax, display_molecules[i], xyz, title, limits)
    fig.supxlabel("Colored: generated molecule   ·   Faint gray: QM9 reference", fontsize=10)
    plt.show()


def check_helpers() -> None:
    """Sanity-check the provided tokenization utilities."""
    assert VOCAB["<PAD>"] == 0
    assert len(VOCAB) == max(VOCAB.values()) + 1
    # Bracket H counts and stereochemistry do not create extra coordinate-bearing atoms.
    token_cases: list[tuple[str, list[int]]] = [
        ("CCO", [0, 1, 2]),
        ("ClCBr", [0, 2, 3]),
        ("[13CH2-][NH3+]", [3, 9]),
        ("c1cc[nH]c1", [0, 2, 3, 5, 8]),
        ("N[C@@H](C)C(=O)O", [0, 2, 8, 10, 13, 15]),
    ]
    for smiles, expected_positions in token_cases:
        tokenized = tokenize(smiles)
        assert tokenized.atom_token_indices == expected_positions
        assert "".join(tokenized.tokens) == smiles
        parsed = Chem.MolFromSmiles(smiles)
        assert parsed is not None
        assert len(tokenized.atom_token_indices) == parsed.GetNumHeavyAtoms()
    print(f"Heavy-atom tokenizer: {len(token_cases)} atom-mapping cases passed.")


def check_geometry(batch: Batch, align_fn: AlignFn, loss_fn: LossFn) -> None:
    """Run the notebook's existing geometry smoke checks through its implementations."""
    x, mask = batch.coords, batch.atom_mask
    r = random_rotations(len(x), x.device)
    y = x @ r + torch.tensor([4.0, -3.0, 2.0])
    assert torch.allclose(torch.linalg.det(r), torch.ones(len(x)), atol=1e-5)
    # assert loss_fn(x, y, mask)[0].max() < 1e-8
    p = torch.randn_like(x, requires_grad=True)
    a = loss_fn(p, x, mask)[0]
    shifted_loss = loss_fn(p, y, mask)[0]
    assert torch.isfinite(shifted_loss).all()
    # assert torch.allclose(a, shifted_loss, atol=2e-5)
    padded = x.clone()
    padded[~mask] = 1000.0
    # assert loss_fn(padded, x, mask)[0].max() < 1e-8
    a.mean().backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0
    assert align_fn(x.clone().requires_grad_(), x, mask).requires_grad
    tetrahedron = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
    )
    reflected = tetrahedron * torch.tensor([-1.0, 1.0, 1.0])
    assert loss_fn(tetrahedron, reflected, torch.ones(1, 4, dtype=torch.bool))[0] > 0.1
    print("Geometry checks passed.")
