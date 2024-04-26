
# Chateando con los libros de Pedagogía y Cultura 📚

Este trabajo tiene por finalidad hacer uso de técnicas de IA para
generación de un "chatbot" que permita hacerle preguntas a los [libros de la colección Pedagogía y Cultura](https://isep-cba.edu.ar/web/coleccion-libros/).

Técnica usada: [*Retrieval-Augmented Generation (RAG)*](https://research.ibm.com/blog/retrieval-augmented-generation-RAG).

# APP Online (sólo para hacer pruebas)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appapppy-6mknyufwrzxhkxzknsewmt.streamlit.app/)

⚠️ Es importante tener en cuenta que la versión demo de la aplicación no está diseñada para escenarios de uso a gran escala, donde múltiples usuarios acceden simultáneamente. Esta versión está destinada exclusivamente para propósitos de prueba y su uso está limitado a un número reducido de usuarios

# Aspectos a tener en cuenta

* La APP está en su fase inicial (alfa).
* Aún necesita varias mejoras en la fase de *"Retrieval"*. Esto implica el procesamiento de documentos y la extracción de información de la base de datos vectorial.
* Debido a esto, es posible que el bot no encuentre la información solicitada o que devuelva resultados "incoherentes".