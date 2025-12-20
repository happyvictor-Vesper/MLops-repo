import streamlit as st
import argparse
import os
from datetime import datetime
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

# Import fungsi dari proyek CatVTON Anda
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# --------------------------------------------------------------------------
# FUNGSI buat backend
# --------------------------------------------------------------------------

def parse_args():
    # Kita tidak menggunakan command-line, tapi fungsi pipeline bergantung pada objek 'args' ini.
    # Jadi, kita buat objek args default secara manual.
    class Args:
        base_model_path = "booksforcharlie/stable-diffusion-inpainting"
        resume_path = "zhengchong/CatVTON"
        output_dir = "resource/demo/output"
        width = 768
        height = 1024
        repaint = False
        allow_tf32 = True
        mixed_precision = "bf16" # Penting untuk performa di GPU Anda

    args = Args()
    return args

def image_grid(imgs, rows, cols):
    # Fungsi helper untuk menggabungkan gambar (digunakan di submit_function)
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# --------------------------------------------------------------------------
# FUNGSI INTI (MODEL DAN PROSES)
# --------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """
    Memuat semua model (pipeline, automasker) sekali saja dan menyimpannya di cache Streamlit.
    """
    args = parse_args()
    
    # Download model-model dari Hugging Face
    repo_path = snapshot_download(repo_id=args.resume_path)
    
    # Muat Pipeline CatVTON
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(args.mixed_precision),
        use_tf32=args.allow_tf32,
        device='cuda'
    )
    
    # Muat AutoMasker
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device='cuda', 
    )
    
    return pipeline, automasker, mask_processor, args

def run_tryon(
    person_image_pil,
    cloth_image_pil,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type,
    pipeline,
    automasker,
    mask_processor,
    args,
    progress_bar # <<< Tambahkan argumen progress_bar
):
    """
    Fungsi inti yang menjalankan proses try-on.
    Sekarang diperbarui untuk mengontrol progress bar.
    """

    # Folder output
    progress_bar.progress(5, text="Memulai... Menyiapkan folder output.")
    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    # Generator seed
    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    # Siapkan gambar input (sudah dalam format PIL)
    progress_bar.progress(10, text="Menyiapkan gambar input...")
    person_image = person_image_pil
    cloth_image = cloth_image_pil
    
    # Ubah ukuran gambar
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Hasilkan mask secara otomatis
    progress_bar.progress(25, text="Membuat mask otomatis...")
    mask = automasker(
        person_image,
        cloth_type
    )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Proses Inference
    progress_bar.progress(50, text="Menjalankan model try-on... (Ini adalah bagian terlama ⏳)")
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    
    # Post-process (membuat gambar perbandingan)
    progress_bar.progress(90, text="Menggabungkan gambar hasil...")
    if show_type == "result only":
        progress_bar.progress(100, text="Selesai!")
        progress_bar.empty() # Hapus progress bar setelah selesai
        return result_image
    
    masked_person = vis_mask(person_image, mask)
    
    if show_type == "input & result":
        conditions = image_grid([person_image, cloth_image], 2, 1)
        condition_width = args.width // 2
    else: # "input & mask & result"
        conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        condition_width = args.width // 3

    width, height = person_image.size
    conditions = conditions.resize((condition_width, height), Image.NEAREST)
    new_result_image = Image.new("RGB", (width + condition_width + 5, height))
    new_result_image.paste(conditions, (0, 0))
    new_result_image.paste(result_image, (condition_width + 5, 0))
    
    progress_bar.progress(100, text="Selesai!")
    progress_bar.empty() # Hapus progress bar setelah selesai
    
    return new_result_image

# --------------------------------------------------------------------------
# UI/UX STREAMLIT UTAMA
# --------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Virtual Try-On")

    # --- Inisialisasi session_state ---
    # Ini penting untuk menyimpan gambar yang dipilih
    if 'person_image_input' not in st.session_state:
        st.session_state.person_image_input = None
    if 'cloth_image_input' not in st.session_state:
        st.session_state.cloth_image_input = None

    # Panggil st.toast di LUAR fungsi cache
    st.toast("Memuat model... Ini mungkin perlu waktu beberapa menit saat pertama kali.")
    
    # Muat model-model (hanya terjadi sekali)
    pipeline, automasker, mask_processor, args = load_models()
    
    # Tampilkan notifikasi setelah model selesai dimuat
    st.toast("Model berhasil dimuat! 🎉")


    # --- UI Sidebar (Kontrol) ---
    with st.sidebar:
        st.title("👕 Pengaturan")
        st.divider()

        st.header("1. Input Gambar")
        
        # Opsi untuk mengunggah file atau menggunakan contoh
        input_method = st.radio("Pilih sumber gambar:", ["Unggah Sendiri", "Gunakan Contoh"], horizontal=True)
        
        root_path = "resource/demo/example"

        if input_method == "Unggah Sendiri":
            person_file = st.file_uploader("Unggah Foto Orang", type=["jpg", "png", "jpeg"])
            cloth_file = st.file_uploader("Unggah Foto Pakaian", type=["jpg", "png", "jpeg"])
            
            if person_file:
                st.session_state.person_image_input = Image.open(person_file).convert("RGB")
            if cloth_file:
                st.session_state.cloth_image_input = Image.open(cloth_file).convert("RGB")
        
        else: # "Gunakan Contoh"
            
            # Membuat daftar file contoh
            person_men_path = os.path.join(root_path, "person", "men")
            person_women_path = os.path.join(root_path, "person", "women")
            cloth_upper_path = os.path.join(root_path, "condition", "upper")
            cloth_overall_path = os.path.join(root_path, "condition", "overall")

            # Mengatasi error jika folder tidak ada
            person_examples = []
            cloth_examples = []
            if os.path.exists(person_men_path):
                person_examples.extend([os.path.join(person_men_path, f) for f in os.listdir(person_men_path)])
            if os.path.exists(person_women_path):
                person_examples.extend([os.path.join(person_women_path, f) for f in os.listdir(person_women_path)])
            if os.path.exists(cloth_upper_path):
                cloth_examples.extend([os.path.join(cloth_upper_path, f) for f in os.listdir(cloth_upper_path)])
            if os.path.exists(cloth_overall_path):
                cloth_examples.extend([os.path.join(cloth_overall_path, f) for f in os.listdir(cloth_overall_path)])

            if not person_examples or not cloth_examples:
                st.error(f"Folder contoh di '{root_path}' tidak ditemukan. Silakan gunakan mode 'Unggah Sendiri'.")
            else:
                tab_person, tab_cloth = st.tabs(["Contoh Orang", "Contoh Pakaian"])
                
                with tab_person:
                    # Tampilkan galeri dalam 4 kolom
                    cols = st.columns(4)
                    for i, p_path in enumerate(person_examples):
                        col = cols[i % 4]
                        with col:
                            # --- PERBAIKAN DI SINI ---
                            st.image(p_path, use_container_width=True, caption=os.path.basename(p_path))
                            # Tombol untuk memilih gambar, menggunakan 'key' unik
                            if st.button("Pilih", key=f"person_{p_path}", use_container_width=True):
                                st.session_state.person_image_input = Image.open(p_path).convert("RGB")
                                st.toast(f"Memilih {os.path.basename(p_path)}")
                
                with tab_cloth:
                    # Tampilkan galeri dalam 4 kolom
                    cols = st.columns(4)
                    for i, c_path in enumerate(cloth_examples):
                        col = cols[i % 4]
                        with col:
                            # --- PERBAIKAN DI SINI ---
                            st.image(c_path, use_container_width=True, caption=os.path.basename(c_path))
                            # Tombol untuk memilih gambar, menggunakan 'key' unik
                            if st.button("Pilih", key=f"cloth_{c_path}", use_container_width=True):
                                st.session_state.cloth_image_input = Image.open(c_path).convert("RGB")
                                st.toast(f"Memilih {os.path.basename(c_path)}")


        st.divider()
        
        # --- Opsi Lanjutan ---
        st.header("2. Opsi Lanjutan")
        with st.expander("Buka untuk mengatur parameter"):
            cloth_type = st.radio(
                "Tipe Pakaian (untuk Auto-Mask)",
                ["upper", "lower", "overall"],
                index=0, # Default ke 'upper'
                horizontal=True
            )
            num_inference_steps = st.slider(
                "Langkah Inference", min_value=10, max_value=100, step=5, value=50
            )
            guidance_scale = st.slider(
                "Kekuatan CFG (Saturasi)", min_value=0.0, max_value=7.5, step=0.5, value=2.5
            )
            seed = st.slider(
                "Seed Acak", min_value=-1, max_value=10000, step=1, value=42
            )
            show_type = st.radio(
                "Tampilkan Hasil",
                ["result only", "input & result", "input & mask & result"],
                index=2, # Default ke 'input & mask & result'
                horizontal=True
            )

    # --- UI Area Utama (Input dan Hasil) ---
    st.title("Virtual Try-On dengan CatVTON")
    st.write("Pilih gambar di panel sebelah kiri dan klik 'Mulai Try-On'.")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Input Orang")
        # Baca dari session_state
        if st.session_state.person_image_input:
            st.image(st.session_state.person_image_input, use_container_width=True)
        else:
            st.info("Silakan pilih foto orang di sidebar.")
            
    with col2:
        st.header("Input Pakaian")
        # Baca dari session_state
        if st.session_state.cloth_image_input:
            st.image(st.session_state.cloth_image_input, use_container_width=True)
        else:
            st.info("Silakan pilih foto pakaian di sidebar.")

    st.divider()

    # --- Tombol Submit dan Area Hasil ---
    if st.button("Mulai Try-On", type="primary", use_container_width=True):
        if st.session_state.person_image_input is None or st.session_state.cloth_image_input is None:
            st.error("Kesalahan: Pastikan foto orang dan foto pakaian sudah terpilih.")
        else:
            # --- FITUR BARU: PROGRESS BAR ---
            # Buat progress bar di sini
            progress_bar = st.progress(0, text="Memulai proses...")

            # Jalankan semua proses
            try:
                result_image = run_tryon(
                    person_image_pil=st.session_state.person_image_input,
                    cloth_image_pil=st.session_state.cloth_image_input,
                    cloth_type=cloth_type,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    show_type=show_type,
                    pipeline=pipeline,
                    automasker=automasker,
                    mask_processor=mask_processor,
                    args=args,
                    progress_bar=progress_bar # <<< Kirim progress bar ke fungsi
                )
                
                # Tampilkan hasil
                st.header("✨ Hasil Try-On")
                st.image(result_image, use_column_width=True)
                st.success("Proses selesai!")
            
            except Exception as e:
                # Bersihkan progress bar jika terjadi error
                progress_bar.empty()
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.error("Ini sering terjadi jika model gagal mendeteksi orang pada gambar. Coba gunakan gambar contoh atau gambar lain yang lebih jelas.")

if __name__ == "__main__":
    main()