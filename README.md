# Hackaton_Panama
Documentacion final del proyecto "Deteccion y prevencion de desastres naturales"

1. Pagina principal


  1.1 Compilar app.py

  
  Este archivo contiene todo el procesamiento y calculo de las imagenes satelitales, las cuales son accedidas por medio de Sentinel Hub. Este programa se encarga de recibir, procesar, guardar y enviar datos, imagenes, archivos GIF y graficas a cada pagina por separado. 

  
  1.2 Abrir el archivo ./templates/index.html

  
  Aqui se contiene la pagina principal, esta pagina se redirecciona a las paginas inundaciones.html e incendios.hmtl

  
  1.3 Abrir inundaciones.html e incendios.html

  
  El proposito de esta pagina es mostrar el analisis hecho por el codigo de python. En este caso, se conecta por medio de un EndPoint, al establecer conexión se puede leer en la consola del navegador un mensaje de "Conexion exitosa", lo que significa que el programa realiza todas sus funciones, produciendo imagenes, graficas y gif´s, los cuales se guardan en la misma ruta del programa, estas imagenes se pueden visualizar en la pagina correspondiente.

2. Pagina de prueba

  1.1 Compilar ./prueba/apprespaldo.py
  
  1.2 Abrir ./prueba/templates/index.html
    El proposito de este codigo es mostrar el funcionamiento del programa principal codificado en python, el cual toma como argumento las coordenadas ingresadas por el ususario, al presionar el boton de "Enviar coordenadas" se envian por medio de un EndPoint y activando todas las funciones de obtencion y procesamiento de imagenes, posteriormente el archivo se encarga de enviar estos resultados para que se muestren en la misma pagina. 

  El archivo apprespaldo.py ya tiene coordenadas predeterminadas, sin embargo, es posible poner coordenadas propias al quitar el comentario den la linea 717 y borrando la lista "coordenadas" de la linea 718.
  
![image](https://github.com/user-attachments/assets/b7db71e4-2c66-404e-9754-7fbc975b22fc)

  Se recomienda usar las siguientes coordenadas, las cuales fueron unas de las mas recurrentes en nuestras pruebas
  
  -> -113.578704
  
  -> 31.29191
  
  -> -113.474889
  
  -> 31.356268



      
