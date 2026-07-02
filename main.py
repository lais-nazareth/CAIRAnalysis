import sys
import os
import cv2
import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, "Seam Carving"))
import seam_carving
sys.path.pop(0)

sys.path.insert(0, os.path.join(BASE_DIR, "Mesh Based"))
import Mesh
sys.path.pop(0)

sys.path.insert(0, os.path.join(BASE_DIR, "Multi Operator"))
import multiop
sys.path.pop(0)



if __name__ == "__main__":
    image_path = "Seam Carving/lele/lele.jpeg"

    img = cv2.imread(image_path)
    img = util.redimensionarImagem(img, limite=1000)


    target_width = int(img.shape[1])
    target_height = int(img.shape[0] + 100)

    choice = None
    while choice != -1:
        print("======= MENU DE OPÇOES =======\n"
            "1  - Seam Carving\n"
            "2  - Mesh Based\n"
            "3  - Multi Operator\n"
            "-1 - Sair\n")

        choice = input("\nDIGITE A OPÇÃO DESEJADA:\n")
        if choice == "1":
            seam_carving.seam_carving(img, target_width, target_height)
        elif choice == "2":
            Mesh.mesh_based(img, target_width, target_height)
        elif choice == "3":
            multiop.multi_operator(img, target_width, target_height)
    print("Programa finalizado!")