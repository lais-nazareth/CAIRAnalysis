import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, "Seam Carving"))
import MySeamCarving
sys.path.pop(0)

sys.path.insert(0, os.path.join(BASE_DIR, "Mesh Based"))
import Mesh
sys.path.pop(0)

if __name__ == "__main__":
    image_path = "Seam Carving/ellie/ellie.jpeg"

    target_width = 400
    target_height = 400
    choice = None
    while choice != -1:
        print("======= MENU DE OPÇOES =======\n"
            "1  - Seam Carving\n"
            "2  - Mesh Based\n"
            "-1 - Sair\n")

        choice = input("\nDIGITE A OPÇÃO DESEJADA:\n")
        if choice == "1":
            MySeamCarving.seam_carving(image_path, target_width, target_height)
        elif choice == "2":
            Mesh.mesh_based(image_path, target_width, target_height)
    
    print("Programa finalizado!")