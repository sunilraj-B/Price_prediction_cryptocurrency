let cryptocurrencies = [];

// Fetch cryptocurrency data from the CoinGecko API
function fetchCryptocurrencyData() {
    fetch('https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1')
        .then(response => response.json())
        .then(data => {
            cryptocurrencies = data;
            displayCryptocurrencies(data);
        })
        .catch(error => console.error('Error:', error));
}

// Display the cryptocurrency data in the search results
function displayCryptocurrencies(data) {
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = '';

    data.forEach(coin => {
        const listItem = document.createElement('li');
        listItem.textContent = coin.name;
        listItem.value = coin.symbol;
        listItem.addEventListener('click', () => {
            fetchCurrencyDetails(coin.id);
        });
        searchResults.appendChild(listItem);
    });
}
// Linear regression prediction model
function predictPrice(historicalPrices) {
    const data = historicalPrices.map((price, index) => [index, price]);
    const result = regression.linear(data);
    const lastDataPoint = data[data.length - 1];
    const predictedPrice = result.predict([lastDataPoint[0] + 1])[1];

    return predictedPrice;
}

// Fetch details of a specific cryptocurrency
function fetchCurrencyDetails(currencyId) {
    Promise.all([
        fetch(`https://api.coingecko.com/api/v3/coins/${currencyId}/market_chart?vs_currency=usd&days=7&interval=daily`),
        fetch(`https://api.coingecko.com/api/v3/coins/${currencyId}`)
    ])
        .then(responses => Promise.all(responses.map(response => response.json())))
        .then(data => {
            const historicalPrices = data[0].prices.map(entry => entry[1]);
            const currentPrice = data[1].market_data.current_price.usd;
            const imageUrl = data[1].image.large;

            const selectedCurrency = document.getElementById('selectedCurrency');
            selectedCurrency.innerHTML = '';

            const coinInfo = document.createElement('div');
            coinInfo.className = 'coin-info';

            const coinImageContainer = document.createElement('div');
            coinImageContainer.className = 'coin-image-container';
            const coinImage = document.createElement('img');
            coinImage.id = 'coinImage';
            coinImage.src = imageUrl;
            coinImage.alt = 'Coin Image';
            coinImageContainer.appendChild(coinImage);

            const currentPriceElement = document.createElement('div');
            currentPriceElement.id = 'currentPrice';
            currentPriceElement.className = 'info-item';
            currentPriceElement.textContent = 'Current Price: $' + currentPrice;

            // const predictedPriceElement = document.createElement('div');
            // predictedPriceElement.id = 'predictedPrice';
            // predictedPriceElement.className = 'info-item';
            // predictedPriceElement.textContent = 'Predicted Price: $' + (currentPrice * 1.1); // Just a sample prediction
            const predictedPriceElement = document.createElement('div');
            predictedPriceElement.id = 'predictedPrice';
            predictedPriceElement.className = 'info-item';
            const predictedPrice = predictPrice(historicalPrices);
            predictedPriceElement.textContent = 'Predicted Price: $' + predictedPrice.toFixed(2);

            coinInfo.appendChild(coinImageContainer);
            coinInfo.appendChild(currentPriceElement);
            coinInfo.appendChild(predictedPriceElement);

            selectedCurrency.appendChild(coinInfo);

            renderChart(historicalPrices);
        })
        .catch(error => console.error('Error:', error));
}

// Process the fetched historical prices data
function processData(data) {
    const prices = data.map(entry => entry[1]);
    return prices;
}

// Render the chart
function renderChart(prices) {
    const options = {
        series: [
            {
                name: 'Price',
                data: prices
            }
        ],
        chart: {
            type: 'line',
            height: 350
        },
        xaxis: {
            type: 'datetime',
            categories: prices.map((_, index) => index)
        }
    };

    const chart = new ApexCharts(document.querySelector('#chartContainer'), options);
    chart.render();
}

// Handle search input
function handleSearchInput() {
    const searchTerm = searchInput.value.toLowerCase();
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = '';

    if (searchTerm.length > 0) {
        const matchedCurrencies = cryptocurrencies.filter(coin => coin.name.toLowerCase().includes(searchTerm));

        matchedCurrencies.forEach(coin => {
            const listItem = document.createElement('li');
            listItem.textContent = coin.name;
            listItem.value = coin.symbol;
            listItem.addEventListener('click', () => {
                fetchCurrencyDetails(coin.id);
            });
            searchResults.appendChild(listItem);
        });

        searchResults.style.display = 'block';
    } else {
        searchResults.style.display = 'none';
    }
}

// Add event listener to the search input
const searchInput = document.getElementById('searchInput');
searchInput.addEventListener('input', handleSearchInput);

// Fetch cryptocurrency data and initialize the dashboard
fetchCryptocurrencyData();
