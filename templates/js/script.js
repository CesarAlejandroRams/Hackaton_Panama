async function fetchData() {
    const url = 'https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json?cloudCover=[0,10]&startDate=2022-06-11T00:00:00Z&completionDate=2022-06-22T23:59:59Z&maxRecords=10';

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Error en la solicitud: ' + response.statusText);
        }
        const data = await response.json();
        displayData(data); // Función para mostrar los datos en la página
    } catch (error) {
        console.error('Error:', error);
    }
}
document.getElementById('searchProduct').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq \'S1A_IW_GRDH_1SDV_20141031T161924_20141031T161949_003076_003856_634E.SAFE\'';

    fetch(url)
        .then(response => response.json())
        .then(data => {
            let output = '<h2>Resultados del Producto:</h2>';
            data.value.forEach(item => {
                output += `<p>${JSON.stringify(item, null, 2)}</p>`;
            });
            document.getElementById('results').innerHTML = output;
        })
        .catch(error => {
            document.getElementById('results').innerHTML = `<p>Error: ${error.message}</p>`;
        });
});

document.getElementById('searchProducts').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq \'SENTINEL-1\' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq \'productType\' and att/OData.CSC.StringAttribute/Value eq \'EW_GRDM_1S\') and ContentDate/Start gt 2022-05-03T00:00:00.000Z and ContentDate/Start lt 2022-05-03T03:00:00.000Z&$orderby=ContentDate/Start desc';

    fetchOData(url);
});

function fetchOData(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            let output = '<h2>Resultados:</h2>';
            if (data.value) {
                data.value.forEach(item => {
                    output += `<p>${JSON.stringify(item, null, 2)}</p>`;
                });
            } else {
                output += `<p>${JSON.stringify(data, null, 2)}</p>`;
            }
            document.getElementById('results').innerHTML = output;
        })
        .catch(error => {
            document.getElementById('results').innerHTML = `<p>Error: ${error.message}</p>`;
        });
}


document.getElementById('orderby').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$orderby=ContentDate/Start';
    fetchOData(url);
});

document.getElementById('top').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$top=100';
    fetchOData(url);
});

document.getElementById('skip').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$skip=50';
    fetchOData(url);
});

document.getElementById('count').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$count=true';
    fetchOData(url);
});

document.getElementById('expand').addEventListener('click', function() {
    const url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$expand=Attributes';
    fetchOData(url);
});

function fetchOData(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            let output = '<h2>Resultados:</h2>';
            if (data.value) {
                data.value.forEach(item => {
                    output += `<p>${JSON.stringify(item, null, 2)}</p>`;
                });
            } else {
                output += `<p>${JSON.stringify(data, null, 2)}</p>`;
            }
            document.getElementById('results').innerHTML = output;
        })
        .catch(error => {
            document.getElementById('results').innerHTML = `<p>Error: ${error.message}</p>`;
        });
}



function displayData(data) {
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = ''; // Limpia el contenido anterior

    // Itera sobre los productos y los muestra
    data.features.forEach(feature => {
        const product = document.createElement('div');
        product.innerHTML = `
            <h3>${feature.properties.title}</h3>
            <p>Fecha de observación: ${feature.properties.beginTime}</p>
            <p>Cloud Cover: ${feature.properties.cloudCover}%</p>
        `;
        predictionsDiv.appendChild(product);
    });
}

// Llama a la función al cargar la página
window.onload = fetchData;