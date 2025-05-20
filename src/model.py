import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBList
import urllib.request
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt

# Функция для вычисления контактной матрицы по координатам атомов CA
def compute_contact_matrix(coords, threshold=8.0):
    """
    Вычисляет матрицу, где на позиции (i, j) стоит 1, если расстояние между остатками i и j меньше threshold.
    """
    if len(coords) == 0:
        return np.array([])
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    contacts = (dists < threshold).astype(np.int32)
    np.fill_diagonal(contacts, 0)
    return contacts

# Преобразование контактной матрицы в edge_index для PyTorch Geometric
def get_edge_index(contacts):
    if contacts.size == 0:
        return torch.tensor([[], []], dtype=torch.long)
    # Получаем индексы ненулевых элементов (контактов)
    src, dst = np.nonzero(contacts)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

# Определение простой графовой нейронной сети для классификации узлов (определения доменов)
class DomainGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, emb_channels = 10, mlp_hidden=64, mlp_layers=2):
        super(DomainGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, emb_channels)
        # Полносвязная часть (MLP)
        mlp = []
        mlp.append(nn.Linear(emb_channels, mlp_hidden))
        mlp.append(nn.ReLU())
        for _ in range(mlp_layers - 1):
            mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(mlp_hidden, num_classes))
        self.mlp = nn.Sequential(*mlp)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = self.mlp(x)
        return x

# Извлечение координат CA и последовательной нумерации из PDB-файла
def extract_ca_coords_and_ids(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    coords = []
    res_ids = []
    for residue in chain:
        # Игнорируем неаминокислотные остатки (например, HETATM или вода)
        if residue.id[0] != " ":
            continue
        if "CA" in residue:
            coords.append(residue["CA"].get_coord())
            res_ids.append(residue.id[1])
    if len(coords) != 0:
        coords = np.vstack(coords)
    else:
        coords = np.array([])
    return coords, res_ids
# Функция для загрузки границ доменов из файла SCOP
# Формат строки: d1a4ya_ 1a4y A:    1-   85 DOMAIN
# Возвращает список кортежей (start, end) для заданного pdb_id и chain

def load_scop_domains(scop_file, max_entries=None):
    """
    Загружает все домены из файла SCOP (scop-cla-latest.txt).
    Возвращает словарь: {(pdb_id, chain_id): [(start, end), ...], ...}
    Использует интервалы из FA-PDBREG (3-я колонка, например G:5-127,G:230-446)
    max_entries — максимальное число считанных интервалов (None — без ограничения)
    """
    domains = {}
    count = 0
    with open(scop_file, "r") as f:
        for line in f:
            if max_entries is not None and count >= max_entries:
                break
            if line.startswith("#") or len(line) < 30:
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pdb_id = parts[1].lower()
            reg = parts[2]  # FA-PDBREG, например G:5-127,G:230-446
            for region in reg.split(","):
                if ":" in region and "-" in region:
                    chain, rng = region.split(":")
                    chain = chain.strip().upper()
                    try:
                        start, end = rng.split("-")
                        key = (pdb_id, chain)
                        if key not in domains:
                            domains[key] = []
                        domains[key].append((int(start), int(end)))
                        count += 1
                    except ValueError:
                        continue
                if max_entries is not None and count >= max_entries:
                    break
            if max_entries is not None and count >= max_entries:
                break
    return domains

def download_scop_file(scop_url, local_path):
    """
    Скачивает файл SCOP по ссылке, если он отсутствует локально.
    """
    if not os.path.exists(local_path):
        print(f"Скачивание SCOP файла из {scop_url}...")
        urllib.request.urlretrieve(scop_url, local_path)
        print("SCOP файл успешно загружен.")
    else:
        print("SCOP файл уже существует.")

def get_domain_mask(res_ids, domain_ranges):
    """
    Возвращает бинарный вектор: 1 — остаток внутри домена, 0 — вне доменов
    """
    mask = np.zeros(len(res_ids), dtype=np.int32)
    for start, end in domain_ranges:
        for i, res in enumerate(res_ids):
            if start <= res <= end:
                mask[i] = 1
    return torch.tensor(mask, dtype=torch.long)

def get_domain_boundary_mask(res_ids, domain_ranges):
    """
    Возвращает бинарный вектор: 1 — граница домена (start или end), 0 — не граница
    """
    mask = np.zeros(len(res_ids), dtype=np.int32)
    for start, end in domain_ranges:
        for i, res in enumerate(res_ids):
            if res == start or res == end:
                mask[i] = 1
    return torch.tensor(mask, dtype=torch.long)
MAX_ENTRIES = 1200  # Максимальное количество интервалов для загрузки из SCOP
NUM_EPOCHS = 300  # Число эпох обучения
def main():
    
    scop_url = "https://www.ebi.ac.uk/pdbe/scop/files/scop-cla-latest.txt"
    scop_file = "./data/scop-cla-latest.txt"
    download_scop_file(scop_url, scop_file)

    # Загружаем все домены из SCOP

    all_domains = load_scop_domains(scop_file, MAX_ENTRIES)
    # Формируем ground truth разметку для всех белков и цепей
    all_training_data = []
    for (pdb, chain), domain_ranges in tqdm(all_domains.items()):
        pdb_file = f"./data/pdb/{pdb}.pdb"
        if not os.path.exists(pdb_file):
            pdbl = PDBList()
            pdb_file_path = pdbl.retrieve_pdb_file(pdb, pdir=".", file_format="pdb")
            # Переименовываем только если файл существует
            if os.path.exists(pdb_file_path):
                os.rename(pdb_file_path, pdb_file)
            else:
                print(f"Файл {pdb_file_path} не найден. Пропускаем {pdb}_{chain}.")
                continue
        coords, res_ids = extract_ca_coords_and_ids(pdb_file)
        if len(coords) == 0 or len(res_ids) == 0:
            continue  # Пропускаем пустые последовательности
        contacts = compute_contact_matrix(coords, threshold=8.0)
        edge_index = get_edge_index(contacts)
        x = torch.tensor(coords, dtype=torch.float)
        y = get_domain_boundary_mask(res_ids, domain_ranges)  # 1 — граница, 0 — не граница
        all_training_data.append((x, edge_index, y, pdb, chain))
    print(f"Загружено белков для обучения: {len(all_training_data)}")
    num_classes = 2
    # Делим на обучающую и тестовую выборку (например, 80/20)
    np.random.seed(42)
    indices = np.arange(len(all_training_data))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    train_data = [all_training_data[i] for i in train_idx]
    test_data = [all_training_data[i] for i in test_idx]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Обучение и оценка
    model_path = 'domain_gcn_model.pth'
    model = DomainGCN(in_channels=3, hidden_channels=32, num_classes=num_classes)
    model = train_model(model, train_data, NUM_EPOCHS, lr=0.01, log_dir="./runs/domain_gcn", model_path=model_path)
    # Загрузка модели с диска перед оценкой
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f'Модель загружена из {model_path}')
    evaluate_model(model, test_data)
    
def train_model(model, train_data, num_epochs, lr=0.01, log_dir=None, model_path='domain_gcn_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    #)
    # Приоритет классу 1 (граница)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 7.]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 6.]))
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    model.train()
    for epoch in tqdm(range(num_epochs)):
        losses = []
        for x, edge_index, y, pdb, chain in train_data:
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(losses)
        if writer:
            writer.add_scalar('Loss/train', mean_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        tqdm.write(f"  Loss: {mean_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(epoch + 1)
    if writer:
        writer.close()
    # Сохраняем модель после обучения
    torch.save(model.state_dict(), model_path)
    print(f'Модель сохранена в {model_path}')
    return model

def domak_predict(coords, res_ids, threshold=8.0):
    """
    DOMAK-like baseline: возвращает бинарную маску границ (1 — граница, 0 — не граница)
    Граница — это позиции разрывов между остатками (расстояние > threshold), а также первый и последний остаток.
    """
    mask = np.zeros(len(coords), dtype=np.int32)
    if len(coords) < 2:
        return torch.zeros(len(coords), dtype=torch.long)
    dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    boundaries = np.where(dists > threshold)[0] + 1  # +1 — правая граница разрыва
    # Отметим первый и последний остаток как границы (по аналогии с ground truth)
    mask[0] = 1
    mask[-1] = 1
    for b in boundaries:
        mask[b] = 1
    return torch.tensor(mask, dtype=torch.long)

def split_predict(coords, n_domains=2):
    """
    SPLIT baseline: равномерно делит последовательность на n_domains доменов.
    Возвращает бинарную маску границ (1 — граница, 0 — не граница).
    Границами считаются первый, последний и равномерно расставленные позиции.
    """
    L = len(coords)
    mask = np.zeros(L, dtype=np.int32)
    if L == 0 or n_domains < 1:
        return torch.zeros(L, dtype=torch.long)
    # Первый и последний всегда границы
    mask[0] = 1
    mask[-1] = 1
    if n_domains == 1:
        return torch.tensor(mask, dtype=torch.long)
    # Внутренние границы
    step = L // n_domains
    for i in range(1, n_domains):
        idx = i * step
        if idx < L-1:
            mask[idx] = 1
    return torch.tensor(mask, dtype=torch.long)

def plot_domain_predictions(x, y_true, y_pred, y_domak, y_split, pdb_id, save_path=None):
    """
    Визуализация разметки доменных границ и сегментов для одной последовательности:
    - y_true: ground truth
    - y_pred: предсказание модели
    - y_domak: DOMAK baseline
    - y_split: SPLIT baseline
    """
    L = len(y_true)
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), gridspec_kw={'height_ratios': [1, 2]})
    # Верхний график: маски границ
    axes[0].step(np.arange(L), y_true, where='mid', label='Ground Truth', lw=3)
    axes[0].step(np.arange(L), y_pred, where='mid', label='GCN', lw=2)
    axes[0].step(np.arange(L), y_domak, where='mid', label='DOMAK', lw=2)
    axes[0].step(np.arange(L), y_split, where='mid', label='SPLIT', lw=2)
    axes[0].set_ylabel('Граница (1/0)')
    axes[0].set_title(f'Границы доменов для {pdb_id}')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlim(0, L-1)
    # Нижний график: сегменты (цветные полосы)
    def plot_segments(ax, mask, label, color):
        segments = mask_to_segments(mask)
        for (start, end) in segments:
            ax.axvspan(start, end, alpha=0.3, color=color, label=label)
    ax2 = axes[1]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    plot_segments(ax2, y_true, 'Ground Truth', colors[0])
    plot_segments(ax2, y_pred, 'GCN', colors[1])
    plot_segments(ax2, y_domak, 'DOMAK', colors[2])
    plot_segments(ax2, y_split, 'SPLIT', colors[3])
    ax2.set_xlim(0, L-1)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel('Индекс остатка')
    ax2.set_title('Сегменты доменов (цветные полосы)')
    # Убираем дублирующиеся легенды
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

    from Bio.PDB import PDBParser
    import nglview as nv
    # Визуализация предсказания на примере
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("PHA-L", f"./data/{id}.pdb")
    view = nv.show_biopython(structure)

def evaluate_model(model, test_data):
    ious, f1s, mbds = [], [], []
    ious_domak, f1s_domak, mbds_domak = [], [], []
    iou_domains, iou_domains_domak = [], []
    ious_split, f1s_split, mbds_split = [], [], []
    iou_domains_split = []
    for x, edge_index, y, pdb, chain in test_data:
        model.eval()
        pred = model(x, edge_index).argmax(dim=1)
        iou = compute_iou(pred, y)
        _, _, f1 = compute_boundary_f1(pred, y, window=5)
        mbd = compute_mean_boundary_deviation(pred, y)
        ious.append(iou)
        f1s.append(f1)
        mbds.append(mbd)
        # IoU для доменов
        iou_dom = compute_iou_domains(pred, y)
        iou_domains.append(iou_dom)
        # DOMAK baseline
        domak_mask = domak_predict(x.cpu().numpy(), np.arange(1, len(x)+1))
        iou_d = compute_iou(domak_mask, y)
        _, _, f1_d = compute_boundary_f1(domak_mask, y, window=5)
        mbd_d = compute_mean_boundary_deviation(domak_mask, y)
        ious_domak.append(iou_d)
        f1s_domak.append(f1_d)
        mbds_domak.append(mbd_d)
        iou_domains_domak.append(compute_iou_domains(domak_mask, y))
        # SPLIT baseline (количество доменов из ground truth)
        n_domains = max(1, len(mask_to_segments(y)))
        split_mask = split_predict(x.cpu().numpy(), n_domains=n_domains)
        iou_s = compute_iou(split_mask, y)
        _, _, f1_s = compute_boundary_f1(split_mask, y, window=5)
        mbd_s = compute_mean_boundary_deviation(split_mask, y)
        ious_split.append(iou_s)
        f1s_split.append(f1_s)
        mbds_split.append(mbd_s)
        iou_domains_split.append(compute_iou_domains(split_mask, y))
    # Формируем таблицу результатов
    results = {
        'Метрика': [
            'IoU (границы)',
            'IoU (домены)',
            'Boundary F1-score',
            'Mean Boundary Deviation'
        ],
        'Model (mean±std)': [
            f"{np.nanmean(ious):.4f} ± {np.nanstd(ious):.4f}",
            f"{np.nanmean(iou_domains):.4f} ± {np.nanstd(iou_domains):.4f}",
            f"{np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}",
            f"{np.nanmean(mbds):.2f} ± {np.nanstd(mbds):.2f}"
        ],
        'DOMAK (mean±std)': [
            f"{np.nanmean(ious_domak):.4f} ± {np.nanstd(ious_domak):.4f}",
            f"{np.nanmean(iou_domains_domak):.4f} ± {np.nanstd(iou_domains_domak):.4f}",
            f"{np.nanmean(f1s_domak):.4f} ± {np.nanstd(f1s_domak):.4f}",
            f"{np.nanmean(mbds_domak):.2f} ± {np.nanstd(mbds_domak):.2f}"
        ],
        'SPLIT (mean±std)': [
            f"{np.nanmean(ious_split):.4f} ± {np.nanstd(ious_split):.4f}",
            f"{np.nanmean(iou_domains_split):.4f} ± {np.nanstd(iou_domains_split):.4f}",
            f"{np.nanmean(f1s_split):.4f} ± {np.nanstd(f1s_split):.4f}",
            f"{np.nanmean(mbds_split):.2f} ± {np.nanstd(mbds_split):.2f}"
        ]
    }
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv('results_metrics.csv', index=False)
    # Визуализация для белка с >=2 доменами
    found = False
    for x, edge_index, y, pdb, chain in test_data:
        n_domains = len(mask_to_segments(y))
        if n_domains >= 3:
            model.eval()
            pred = model(x, edge_index).argmax(dim=1)
            domak_mask = domak_predict(x.cpu().numpy(), np.arange(1, len(x)+1))
            split_mask = split_predict(x.cpu().numpy(), n_domains=n_domains)
            plot_domain_predictions(
                x, y.cpu().numpy(), pred.cpu().numpy(), domak_mask.cpu().numpy(), split_mask.cpu().numpy(),
                pdb_id=f"{pdb}_{chain}", save_path="example_pred.png"
            )
            print(f"Визуализация сохранена в example_pred.png для {pdb}_{chain} (домены: {n_domains})")
            found = True
            break
    if not found:
        print("В тестовой выборке нет белков с двумя и более доменами для визуализации.")
    return ious, f1s, mbds

def compute_iou(y_pred, y_true):
    """
    IoU для бинарных масок (1 — граница домена, 0 — не граница)
    """
    intersection = ((y_pred == 1) & (y_true == 1)).sum().item()
    union = ((y_pred == 1) | (y_true == 1)).sum().item()
    if union == 0:
        return float('nan')
    return intersection / union

def mask_to_segments(mask, min_length=20):
    """
    Преобразует бинарную маску границ (1 — граница, 0 — не граница) в список сегментов (start, end),
    игнорируя короткие сегменты (домен считается, только если длина >= min_length)
    """
    mask = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
    boundaries = np.where(mask == 1)[0]
    segments = []
    prev = 0
    for b in boundaries:
        if b > prev:
            if b - prev >= min_length:
                segments.append((prev, b-1))
        prev = b
    if prev < len(mask):
        if len(mask) - prev >= min_length:
            segments.append((prev, len(mask)-1))
    return segments

def compute_iou_domains(pred_mask, true_mask, min_length=20):
    """
    IoU для доменных сегментов, полученных из бинарных масок границ,
    игнорируя короткие сегменты (домен считается, только если длина >= min_length)
    """
    pred_segments = mask_to_segments(pred_mask, min_length=min_length)
    true_segments = mask_to_segments(true_mask, min_length=min_length)
    ious = []
    for ps in pred_segments:
        best_iou = 0
        for ts in true_segments:
            set_ps = set(range(ps[0], ps[1]+1))
            set_ts = set(range(ts[0], ts[1]+1))
            inter = len(set_ps & set_ts)
            union = len(set_ps | set_ts)
            if union > 0:
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
        ious.append(best_iou)
    if len(ious) == 0:
        return float('nan')
    return np.mean(ious)

def get_boundaries_from_mask(mask):
    """
    Возвращает индексы границ (позиции, где mask==1)
    """
    mask = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
    return np.where(mask == 1)[0]

def compute_boundary_f1(pred_mask, true_mask, window=5):
    """
    Boundary F1-score: сравнивает найденные и истинные границы с допуском window
    """
    pred_bound = get_boundaries_from_mask(pred_mask)
    true_bound = get_boundaries_from_mask(true_mask)
    if len(pred_bound) == 0 or len(true_bound) == 0:
        return float('nan'), float('nan'), float('nan')
    matched_pred = np.zeros(len(pred_bound), dtype=bool)
    matched_true = np.zeros(len(true_bound), dtype=bool)
    for i, pb in enumerate(pred_bound):
        for j, tb in enumerate(true_bound):
            if abs(pb - tb) <= window:
                matched_pred[i] = True
                matched_true[j] = True
    precision = matched_pred.sum() / len(pred_bound) if len(pred_bound) > 0 else 0
    recall = matched_true.sum() / len(true_bound) if len(true_bound) > 0 else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def compute_mean_boundary_deviation(pred_mask, true_mask):
    """
    Mean Boundary Deviation (MBD): среднее абсолютное отклонение между ближайшими границами
    """
    pred_bound = get_boundaries_from_mask(pred_mask)
    true_bound = get_boundaries_from_mask(true_mask)
    if len(pred_bound) == 0 or len(true_bound) == 0:
        return float('nan')
    dists = []
    for tb in true_bound:
        dists.append(np.min(np.abs(pred_bound - tb)))
    return np.mean(dists)

if __name__ == "__main__":
    main()