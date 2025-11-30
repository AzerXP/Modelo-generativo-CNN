import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import GPT2TokenizerFast

# ==========================
# 1. Tokenizer GPT-2 byte-level
# ==========================
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Añadir tokens especiales
special_tokens = {
    "additional_special_tokens": ["<start>", "<end>", "<unk>"],
    "pad_token": "<pad>"
}
tokenizer.add_special_tokens(special_tokens)

vocab_size = len(tokenizer)

# ==========================
# 2. Dataset con tokenizer
# ==========================
class DiagramDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=200):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.max_len = max_len

    def _load_json_tokens(self, img_name):
        json_name = img_name.replace(".png", ".json")
        json_path = os.path.join(self.root_dir, json_name)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.dumps(json.load(f), ensure_ascii=False)

            # Añadir tokens especiales al inicio y fin
            text = "<start> " + data + " <end>"

            tokens = tokenizer.encode(
                text,
                max_length=self.max_len,
                truncation=True,
                padding="max_length"
            )
            return torch.tensor(tokens)
        else:
            return torch.tensor([tokenizer.pad_token_id] * self.max_len)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        tokens = self._load_json_tokens(img_name)
        return image, tokens

# ==========================
# 3. Modelo Encoder + Transformer Decoder
# ==========================
class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(64*28*28, embed_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, num_layers, vocab_size, max_len=200):
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        seq_len = captions.size(1)
        positions = torch.arange(0, seq_len, device=captions.device).unsqueeze(0)
        embeddings = self.embed(captions) + self.pos_embed(positions)
        memory = features.unsqueeze(0)  # (1, batch, embed_size)
        tgt = embeddings.transpose(0,1) # (seq_len, batch, embed_size)
        out = self.transformer_decoder(tgt, memory)
        out = out.transpose(0,1)        # (batch, seq_len, embed_size)
        return self.fc(out)

class Image2JSON(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, num_layers, vocab_size):
        super(Image2JSON, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = TransformerDecoder(embed_size, num_heads, hidden_size, num_layers, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ==========================
# 4. Entrenamiento
# ==========================
root_dir = "dataset"
transform = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor()])
dataset = DiagramDataset(root_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size, hidden_size, num_heads, num_layers = 256, 512, 8, 4
model = Image2JSON(embed_size, num_heads, hidden_size, num_layers, vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, captions in train_loader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(images, captions[:,:-1])
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:,1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "diagram_image2json.pth")
tokenizer.save_pretrained("./tokenizer")
print("✅ Modelo guardado como diagram_image2json.pth y tokenizer en ./tokenizer")


# ==========================
# 5. Inferencia paso a paso
# ==========================
def limpiar_y_reconstruir(texto):
    """
    Limpia texto generado y reconstruye un JSON mínimo
    con claves: sistema, actores, casos_uso, relaciones.
    Detecta variaciones y errores comunes.
    """
    # Normalizar comillas y espacios
    s = texto.replace("’","\"").replace("“","\"").replace("”","\"").replace("'", "\"")
    s = re.sub(r'\s+', ' ', s)

    # Buscar sistema (más flexible)
    sistema = None
    m = re.search(r'(Sistema|Plataforma|Gestión)\s+([A-Za-z0-9_ÁÉÍÓÚáéíóúñÑ-]+)', s, re.IGNORECASE)
    if m:
        sistema = m.group(0)  # captura toda la frase

    # Actores conocidos (con variaciones)
    actores_patrones = [
        "Cliente", "Administrador", "Usuario", "Supervisor", "Analista", "Empleado", "Gerente"
    ]
    actores = []
    for a in actores_patrones:
        if re.search(a, s, re.IGNORECASE):
            actores.append(a)

    # Casos de uso conocidos (con variaciones y errores comunes)
    casos_patrones = [
        "Registrar usuario", "Autenticar usuario", "Realizar operación", "Consultar datos",
        "Generar reporte", "Gestionar perfil", "Configurar sistema", "Auditar actividad",
        "Actualizar perfil", "Procesar pago", "Emitir factura", "Reservar cita"
    ]
    casos = []
    for c in casos_patrones:
        # Buscar tanto exacto como variaciones (ej: "registrar usuari", "autenticar usu")
        base = c.split()[0]  # primera palabra
        if re.search(base, s, re.IGNORECASE):
            casos.append(c)

    # Reconstruir relaciones simples actor–caso
    relaciones = []
    for a in actores:
        for c in casos:
            if re.search(rf'{a}.*{c}|{c}.*{a}', s, re.IGNORECASE):
                relaciones.append({"actor": a, "caso_uso": c})

    # Si encontramos algo, devolvemos estructura mínima
    if sistema or actores or casos or relaciones:
        return {
            "sistema": sistema or "Sistema desconocido",
            "actores": actores,
            "casos_uso": casos,
            "relaciones": relaciones
        }

    # Si no encontramos nada, devolvemos texto crudo
    return {"flujo_generado": texto}


def infer_image(img_path, model, tokenizer, device, max_len=200, temperature=1.0, top_k=50):
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor()
    ])
    image = Image.open(img_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    features = model.encoder(image)
    start_id = tokenizer.convert_tokens_to_ids("<start>")
    end_id   = tokenizer.convert_tokens_to_ids("<end>")
    token = torch.tensor([start_id]).to(device)

    generated = []

    for pos in range(max_len):
        positions = torch.arange(0, pos+1, device=device).unsqueeze(0)
        embeddings = model.decoder.embed(token.unsqueeze(0)) + model.decoder.pos_embed(positions)
        tgt = embeddings.transpose(0,1)
        memory = features.unsqueeze(0)
        out = model.decoder.transformer_decoder(tgt, memory)
        logits = model.decoder.fc(out[-1])  # último paso

        probs = torch.softmax(logits / temperature, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=top_k)
        sample_idx = torch.multinomial(topk_probs[0], 1).item()
        token_id = topk_idx[0, sample_idx].item()

        if token_id == end_id:
            break
        generated.append(token_id)
        token = torch.tensor([token_id]).to(device)

    # Texto crudo generado
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print("\n🟡 Texto crudo generado por el modelo:")
    print(text)

    # Intentar parsear como JSON
    if not text.startswith("{"):
        text = "{ " + text
    if not text.endswith("}"):
        text = text + " }"

    try:
        flujo = json.loads(text)
    except:
        flujo = limpiar_y_reconstruir(text)

    return flujo

# ==========================
# 6. Narrativa desde JSON
# ==========================
def flujo_a_texto(flujo_json):
    partes = []

    # Caso: texto crudo sin parsear
    if "flujo_generado" in flujo_json:
        return f"Descripción generada (texto sin estructura): {flujo_json['flujo_generado']}"

    # Caso: JSON válido o reconstruido
    sistema = flujo_json.get("sistema", "Sistema desconocido")
    partes.append(f"El sistema es {sistema}.")

    # Actores
    actores = flujo_json.get("actores", [])
    if actores:
        # Si actores son dicts con 'nombre'
        if isinstance(actores[0], dict):
            nombres = [a.get("nombre", "") for a in actores]
        else:
            nombres = actores
        partes.append("Los actores principales son: " + ", ".join(nombres) + ".")

    # Casos de uso
    casos = flujo_json.get("casos_uso", [])
    if casos:
        if isinstance(casos[0], dict):
            nombres = [c.get("nombre", "") for c in casos]
        else:
            nombres = casos
        partes.append("Los casos de uso incluyen: " + ", ".join(nombres) + ".")

    # Relaciones
    relaciones = flujo_json.get("relaciones", [])
    for rel in relaciones:
        if "actor" in rel and "caso_uso" in rel:
            partes.append(f"El actor {rel['actor']} participa en el caso de uso {rel['caso_uso']}.")
        elif rel.get("tipo") == "include":
            partes.append(f"El caso de uso {rel['origen']} incluye a {rel['destino']}.")
        elif rel.get("tipo") == "extend":
            partes.append(f"El caso de uso {rel['origen']} extiende a {rel['destino']}.")

    return " ".join(partes) if partes else "No se pudo generar narrativa."


# ==========================
# 7. Probar inferencia
# ==========================
tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

model = Image2JSON(embed_size, num_heads, hidden_size, num_layers, len(tokenizer)).to(device)
model.load_state_dict(torch.load("diagram_image2json.pth", map_location=device))
model.eval()  # modo evaluación

# Imagen de prueba
img_test = "dataset/diagrama_0001.png"  # asegúrate de que exista este archivo
flujo = infer_image(img_test, model, tokenizer, device)

# Mostrar resultados
print("🔎 Flujo inferido desde la imagen (JSON):")
print(json.dumps(flujo, indent=2, ensure_ascii=False))

print("\n📝 Narrativa generada:")
print(flujo_a_texto(flujo))


