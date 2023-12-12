import cv2
import os
# import re


def import_images(folder_path):
    # Buat list kosong buat penampung images dan label images
    images = []
    labels = []

    # Memeriksa apakah folder tersebut ada
    if os.path.exists(folder_path):
        # Mendapatkan daftar file dalam folder
        files = os.listdir(folder_path)
        # Loop melalui setiap file dalam folder
        for file_name in files:
            # Memeriksa apakah file tersebut adalah file gambar (misalnya, dengan ekstensi .jpg atau .png)
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                # Menggabungkan path lengkap ke file gambar
                image_path = os.path.join(folder_path, file_name)
                file_name = os.path.basename(image_path)

                """
                #############################################################################
                Kode di bawah digunakan untuk mencari label kelas berdasarkan nama foldernya
                #############################################################################
                """
                labels.append(os.path.basename(folder_path))

                """
                #############################################################################
                Kode di bawah digunakan untuk read image dan convert ke dalam grayscale
                #############################################################################
                """
                # Membaca gambar menggunakan OpenCV
                image = cv2.imread(image_path)
                # Convert ke grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(gray)

            else:
                print(f"Ignoring non-image file: {file_name}")

    else:
        print(f"Folder not found: {folder_path}")

    print("sum of images: " + str(len(images)))
    print("sum of labels: " + str(len(labels)))

    return images, labels
