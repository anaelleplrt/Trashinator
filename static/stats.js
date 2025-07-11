fetch('/api/stats')
    .then(res => res.json())
    .then(data => {
        const annotationData = Object.values(data.annotations);
        const total = annotationData.reduce((a, b) => a + b, 0);
        document.getElementById("total").textContent = data.total_images;

        // 1. Donut pleine/vide
        new Chart(document.getElementById('chartAnnotation'), {
            type: 'doughnut',
            data: {
                labels: Object.keys(data.annotations),
                datasets: [{
                    data: annotationData,
                    backgroundColor: ['#0d6efd', '#ffc107']
                }]
            },
            options: {
                plugins: {
                    legend: { position: 'top' },
                    datalabels: {
                        color: '#fff',
                        formatter: (value) => `${((value / total) * 100).toFixed(1)}%`,
                        font: { weight: 'bold', size: 14 }
                    }
                }
            },
            plugins: [ChartDataLabels]
        });

        // 2. Distribution tailles
        new Chart(document.getElementById('chartFilesize'), {
            type: 'bar',
            data: {
                labels: Object.keys(data.file_size_histogram),
                datasets: [{
                    label: 'Taille (Ko)',
                    data: Object.values(data.file_size_histogram),
                    backgroundColor: '#20c997'
                }]
            }
        });

        // 3. Taux pleines par date
        const dateData = data.date_distribution;
        const labelsDates = Object.keys(dateData);
        const labels = labelsDates.map(d => d.split("T")[0]);
        const pleines = labelsDates.map(d => dateData[d]['pleine'] || 0);
        const vides = labelsDates.map(d => dateData[d]['vide'] || 0);

        new Chart(document.getElementById('chartDates'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Pleine', data: pleines, backgroundColor: '#dc3545' },
                    { label: 'Vide', data: vides, backgroundColor: '#198754' }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    title: { display: true, text: 'Taux de poubelles pleines par date' }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1 }
                    }
                }
            }
        });

        // 4. Répartition par ville (camembert)
        new Chart(document.getElementById('chartPays'), {
            type: 'pie',
            data: {
                labels: Object.keys(data.par_pays),
                datasets: [{
                    data: Object.values(data.par_pays),
                    backgroundColor: [
                        '#0d6efd', '#198754', '#ffc107', '#dc3545', '#20c997',
                        '#6f42c1', '#fd7e14', '#0dcaf0', '#6c757d', '#e83e8c'
                    ]
                }]
            },
            options: {
                plugins: {
                    legend: { position: 'bottom' },
                    title: { display: true, text: 'Répartition par pays' }
                }
            }
        });




        // 6. Dépôts par jour (courbe)
        const depotsValues = labelsDates.map(d => (dateData[d]['pleine'] || 0) + (dateData[d]['vide'] || 0));
        new Chart(document.getElementById('dailyChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Nombre de dépôts',
                    data: depotsValues,
                    fill: false,
                    borderColor: '#0d6efd',
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // 7. Initialisation carte
        const map = L.map('map').setView([0, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);

        // 8. Icônes
        const redIcon = L.icon({
            iconUrl: "https://maps.gstatic.com/mapfiles/ms2/micons/red-dot.png",
            iconSize: [32, 32],
            iconAnchor: [16, 32],
            popupAnchor: [0, -30]
        });
        const greenIcon = L.icon({
            iconUrl: "https://maps.gstatic.com/mapfiles/ms2/micons/green-dot.png",
            iconSize: [32, 32],
            iconAnchor: [16, 32],
            popupAnchor: [0, -30]
        });

        // 9. Marqueurs et filtres
        let markerGroup = L.layerGroup().addTo(map);
        let allMarkers = [];

        function updateMarkers(filter = "all") {
            fetch("/api/gps_coordinates")
                .then(res => res.json())
                .then(locations => {
                    markerGroup.clearLayers();
                    allMarkers = [];
                    const bounds = [];

                    locations.forEach(loc => {
                        if (filter !== "all" && loc.annotation !== filter) return;
                        const icon = loc.annotation === "pleine" ? redIcon : greenIcon;
                        const marker = L.marker([loc.latitude, loc.longitude], { icon })
                            .bindPopup(`📷 ${loc.filename}<br>🗑️ ${loc.annotation}<br>📅 ${loc.upload_date}`);
                        markerGroup.addLayer(marker);
                        allMarkers.push(marker);
                        bounds.push([loc.latitude, loc.longitude]);
                    });

                    document.getElementById("countDisplay").textContent =
                        `${allMarkers.length} poubelle${allMarkers.length > 1 ? 's' : ''} affichée${allMarkers.length > 1 ? 's' : ''}`;

                    if (bounds.length > 0) map.fitBounds(bounds, { padding: [30, 30] });
                });
        }

        // 10. Initialisation des marqueurs
        updateMarkers();

        // 11. Filtres d'annotations
        document.getElementById("filterSelect").addEventListener("change", e => {
            updateMarkers(e.target.value);
        });

        // 12. Centrage manuel
        document.getElementById("fitMapBtn").addEventListener("click", () => {
            const group = new L.featureGroup(allMarkers);
            if (allMarkers.length > 0) map.fitBounds(group.getBounds(), { padding: [30, 30] });
        });

        // 13. Export CSV
        document.getElementById("exportBtn").addEventListener("click", () => {
            if (allMarkers.length === 0) return;
            let csv = "filename,annotation,upload_date,latitude,longitude\n";
            allMarkers.forEach(marker => {
                const content = marker.getPopup().getContent();
                const match = content.match(/📷 (.*?)<br>🗑️ (.*?)<br>📅 (.*?)$/);
                if (!match) return;
                const [_, filename, annotation, date] = match;
                const { lat, lng } = marker.getLatLng();
                csv += `${filename},${annotation},${date},${lat},${lng}\n`;
            });
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "donnees_poubelles.csv";
            a.click();
            URL.revokeObjectURL(url);
        });
    })
    .catch(err => console.error('Erreur de chargement des stats :', err));