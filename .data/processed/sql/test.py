from db import init_db
from crud import *

# Inicializar la base de datos (crea las tablas)
init_db()

# Insertar un nuevo libro
#nuevo_libro = post_book(
#    tittle="Cien años de soledad",
#    author="Gabriel García Márquez",
#    gender="Realismo mágico",
#    description="Novela que narra la historia de la familia Buendía en Macondo.",
#    path="/home/miguel/Escritorio/Programs/OlivIA-RAG/ElDragon.pdf"
#)

# Buscar libros
libros = get_all_books(book_author="García Márquez")
for libro in libros:
    print(libro.Tittle, libro.Author)

# Actualizar un libro
#libro_actualizado = actualizar_libro(
#    libro_id=nuevo_libro.id,
#    descripcion="Nueva descripción más detallada..."
#)

# Eliminar un libro
#if eliminar_libro(libro_id=1):
#    print("Libro eliminado correctamente")