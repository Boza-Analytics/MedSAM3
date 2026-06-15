# Hugging Face setup (required — the model weights are gated)

MedSAM3 = Meta's **SAM3** base model + a **LoRA adapter** trained by the paper authors.
You must download **both**, and the SAM3 base is *gated* (Meta requires you to accept a
license before download). This is a one-time ~10 minute setup.

## 1. Create a Hugging Face account
- Go to https://huggingface.co/join and sign up (free).

## 2. Accept the SAM3 license (the gated base model)
- Visit https://huggingface.co/facebook/sam3 while logged in.
- You'll see a box like *"You need to agree to share your contact information to access this model."*
- Fill the short form and click **Agree and access repository**.
- Access is usually granted instantly. If it says "pending", wait until the page shows the
  file list (Files tab) before continuing — downloads fail until you're approved.

## 3. Check the MedSAM3 LoRA weights repo
- Visit https://huggingface.co/lal-Joey/MedSAM3_v1
- Note the filename of the `.pt` / `.safetensors` LoRA checkpoint under the **Files** tab.
  (The setup script downloads the whole repo, so you don't strictly need the exact name,
  but it's good to know what you got.)

## 4. Create an access token
- Go to https://huggingface.co/settings/tokens
- Click **Create new token** → type **Read** → name it e.g. `medsam3` → **Create**.
- **Copy the token now** (starts with `hf_...`). You won't be able to see it again.

## 5. Log in on the machine that will run the model (your EC2 box)
```bash
source ~/medsam3-venv/bin/activate
huggingface-cli login        # paste your hf_... token when prompted
#   (newer CLI: `hf auth login` works too)
```
Verify it worked:
```bash
huggingface-cli whoami
```

That's it. The inference script pulls the SAM3 base automatically (`load_from_HF=True`),
and `setup_ec2.sh` downloads the MedSAM3 LoRA into `weights/medsam3_v1/`.
