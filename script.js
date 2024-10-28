// URL para la solicitud a la API STAC de Copernicus (Sentinel-1, coordenadas bbox dadas)
const apiURL = 'https://catalogue.dataspace.copernicus.eu/stac/collections/SENTINEL-1/items?bbox=-89.936487,13.10503,-88.596674,14.153738&datetime=2024-08-03T00:00:00Z/2024-08-20T23:59:59Z';

// Función para obtener los datos de la API
async function obtenerDatos() {
    try {
        // Hacer la solicitud HTTP a la API
        const response = await fetch(apiURL);
        // Verificamos que la solicitud fue exitosa
        if (!response.ok) {
            throw new Error(`Error al obtener datos: ${response.status}`);
        }

        // Convertimos la respuesta a JSON
        const data = await response.json();

        // Llamamos a la función para mostrar los datos
        mostrarDatos(data);
    } catch (error) {
        // En caso de error, lo mostramos en la consola y en la página
        console.error(error);
        document.getElementById('resultados').innerHTML = `<p>Error: ${error.message}</p>`;
    }
}

// Función para mostrar los datos en el HTML
function mostrarDatos(data) {
    const contenedor = document.getElementById('resultados');

    // Si no hay productos en los datos obtenidos
    if (data.features.length === 0) {
        contenedor.innerHTML = '<p>No se encontraron productos para esta área.</p>';
        return;
    }

    // Limpiamos el contenido anterior
    contenedor.innerHTML = '';

    // Iteramos sobre los productos (features) obtenidos
    data.features.forEach((feature, index) => {
        // Creamos un elemento para mostrar la información de cada producto
        const productDiv = document.createElement('div');
        productDiv.innerHTML = `
            <h3>Producto ${index + 1}</h3>
            <p><strong>ID:</strong> ${feature.id}</p>
            <p><strong>Fecha:</strong> ${feature.properties.datetime}</p>
            <p><strong>Plataforma:</strong> ${feature.properties.platform}</p>
            <p><strong>Misión:</strong> ${feature.collection}</p>
        `;
        contenedor.appendChild(productDiv);
    });
}

// Llamamos a la función para obtener los datos al cargar la página
obtenerDatos();
