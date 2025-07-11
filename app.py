from imports import*


#charge le modèle YOLO pour la détection d'objets
model_yolo = YOLO("crop/runs/detect_poubelle/weights/best.pt")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DB_FILE = 'app.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/contours", exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ===== CROP ======

def crop_poubelle(image_path):
    crop_folder = "static/uploads/detect_poubelle/crops/poubelle"

    # Récupérer liste avant prédiction
    before_files = set(os.listdir(crop_folder)) if os.path.exists(crop_folder) else set()

    # YOLO
    results = model_yolo.predict(
        source=image_path,
        save=True,
        save_crop=True,
        project="static/uploads",
        name="detect_poubelle",
        exist_ok=True
    )

    # Liste après prédiction
    after_files = set(os.listdir(crop_folder)) if os.path.exists(crop_folder) else set()

    # Nouveaux fichiers = fichiers ajoutés
    new_files = after_files - before_files

    if new_files:
        # Si plusieurs fichiers, on prend le plus récent
        new_file = max(new_files, key=lambda x: os.path.getmtime(os.path.join(crop_folder, x)))
        crop_path = os.path.join(crop_folder, new_file)
        return crop_path

    # Aucun nouveau fichier → pas de crop
    return None

# ===== BDD ======

def init_db():

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                path TEXT,
                upload_date TEXT,
                annotation TEXT,
                ratio_light, ratio_dark, edge_density, mean_r, mean_g, mean_b, num_blobs,ratio_white, top_edge_density, mean_saturation,
                file_size_kb, latitude, longitude,
                histogram TEXT
            )
        ''')
        conn.commit()

init_db()


# === Features ===

def get_image_features(image_path):
    # Ouvrir image
    img = cv2.imread(image_path)
    if img is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # Redimensionner plus petit pour traitement rapide
    img_resized = cv2.resize(img, (200, 200))

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Créer les masques
    light_mask = (gray > 180).astype(np.uint8)
    dark_mask = (gray < 50).astype(np.uint8)

    light_pixels = cv2.countNonZero(light_mask)
    dark_pixels = cv2.countNonZero(dark_mask)
    total_pixels = gray.shape[0] * gray.shape[1]

    ratio_light = light_pixels / total_pixels
    ratio_dark = dark_pixels / total_pixels

    # Contours (pour "désordre")
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels

    # Nombre de blobs
    num_labels, labels_im = cv2.connectedComponents(edges)
    num_blobs = num_labels - 1  # on enlève le fond

    # Moyenne R, G, B
    mean_b = np.mean(img_resized[:, :, 0])
    mean_g = np.mean(img_resized[:, :, 1])
    mean_r = np.mean(img_resized[:, :, 2])

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    white_mask = (img_rgb[:, :, 0] > 220) & (img_rgb[:, :, 1] > 220) & (img_rgb[:, :, 2] > 220)
    num_white_pixels = np.sum(white_mask)
    ratio_white = num_white_pixels / total_pixels

    return (ratio_light, ratio_dark, edge_density, mean_r, mean_g, mean_b, num_blobs, ratio_white)


def get_additional_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0.0

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Découpe haut de l'image
    height = gray.shape[0]
    top = gray[:height // 3, :]
    
    # Edge density en haut
    edges_top = cv2.Canny(top, 50, 150)
    top_edge_density = np.sum(edges_top > 0) / (top.shape[0] * top.shape[1])
    
    # Saturation moyenne
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mean_saturation = np.mean(sat)
    
    return top_edge_density, mean_saturation


# === Classification ===

def classify_image_auto(ratio_light, ratio_dark, edge_density, mean_r, mean_g, mean_b, num_blobs, ratio_white, top_edge_density, mean_saturation):

    max_diff = max(abs(mean_r - mean_g), abs(mean_r - mean_b), abs(mean_g - mean_b))

    if edge_density > 0.17 and ratio_dark > 0.2 and num_blobs > 100 and ratio_white > 0:
        return "pleine"

    if edge_density > 0.17 and ratio_dark > 0.2 and mean_saturation > 35:
        return "pleine"
    
    if num_blobs > 100 and ratio_dark > 0.3 and edge_density > 0.1:
        return "pleine"
    
    if num_blobs > 68 and ratio_dark > 0.13 and edge_density > 0.08 and edge_density < 0.115 and mean_saturation > 45 and mean_g > 75 and ratio_light < 0.15:
        return "pleine"

    print("ratio_white ici : !!!!!", ratio_white)
    if max_diff> 8 and ratio_white > 0.007 and edge_density > 0.17 and ratio_dark > 0.05 and top_edge_density > 0.25 and mean_saturation > 68:
        return "pleine"
    
    if mean_r > 95 and mean_saturation > 35 and mean_g > 100 and mean_b > 85.76 and ratio_white > 0.00029 and top_edge_density > 0.07 and ratio_light < 0.26 and ratio_light > 0.0026:
        return "pleine"

    if edge_density > 0.17 and num_blobs > 135 and ratio_white > 0.02 and ratio_light < 0.3213 and ratio_light > 0.06:
        return "pleine"
    
    # Si clarté et peu de contours → vide
    if ratio_light < 0.1 and edge_density < 0.15:
        return "vide"
    
    # Cas par défaut
    return "vide"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# == Fonction pour calculer l'histogramme pour ensuite lafficher dans le tableau ==
def compute_color_histogram(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    chans = cv2.split(img)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [16], [0, 256])
        hist = hist.flatten()
        features.extend(hist.tolist())
    return json.dumps(features)  # liste de 48 éléments (16 R, 16 G, 16 B)




# ====================== Routes ======================


# ====================== Routes page d'accueil ======================

@app.route('/', methods=['GET', 'POST'])
def index():
    annotation_label = None
    image_url = None

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Croper la poubelle automatiquement
            crop_path = crop_poubelle(filepath)

            if crop_path and os.path.exists(crop_path):
                image_to_analyze = crop_path
                image_url = f"/{crop_path.replace(os.sep, '/')}"
            else:
                image_to_analyze = filepath
                image_url = f"/{filepath.replace(os.sep, '/')}"

            # Extraire les features depuis le crop ou l'image d'origine
            ratio_light, ratio_dark, edge_density, mean_r, mean_g, mean_b, num_blobs, ratio_white = get_image_features(image_to_analyze)
            top_edge_density, mean_saturation = get_additional_features(image_to_analyze)
            file_size_kb = os.path.getsize(image_to_analyze) / 1024  # taille en Ko

            # Annotation automatique
            annotation_label = classify_image_auto(
                ratio_light, ratio_dark, edge_density,
                mean_r, mean_g, mean_b, num_blobs,
                ratio_white, top_edge_density, mean_saturation
            )
            
            cities = [
                ("São Paulo", -23.55, -46.63),
                ("Buenos Aires", -34.60, -58.38),
                ("Mexico City", 19.43, -99.13),
                ("Bogotá", 4.71, -74.07),
                ("Lima", -12.04, -77.03),
                ("Santiago", -33.45, -70.66),
                ("Caracas", 10.48, -66.88),
                ("Quito", -0.18, -78.47),
                ("Montevideo", -34.90, -56.19),
                ("La Paz", -16.5, -68.15)
            ]

            city, base_lat, base_lon = choice(cities)
            latitude = round(base_lat + uniform(-0.2, 0.2), 6)
            longitude = round(base_lon + uniform(-0.2, 0.2), 6)
          

            # Enregistrer dans la BDD
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                histogram_json = compute_color_histogram(image_to_analyze)
                cursor.execute('''
                    INSERT INTO images (
                        filename, path, upload_date, annotation,
                        ratio_light, ratio_dark, edge_density,
                        mean_r, mean_g, mean_b, num_blobs, ratio_white,
                        top_edge_density, mean_saturation, file_size_kb,
                        latitude, longitude, histogram
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    os.path.relpath(image_to_analyze, start=UPLOAD_FOLDER).replace("\\", "/"),
                    image_to_analyze,
                    datetime.now().isoformat(),
                    annotation_label,
                    ratio_light, ratio_dark, edge_density,
                    mean_r, mean_g, mean_b,
                    num_blobs, ratio_white,
                    top_edge_density, mean_saturation,
                    file_size_kb, latitude, longitude,
                    histogram_json 
                ))

                conn.commit()
    
    # Charger toutes les images
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
    SELECT id, filename, annotation, upload_date,
           ratio_light, ratio_dark, edge_density,
           mean_r, mean_g, mean_b, num_blobs, ratio_white,
           top_edge_density, mean_saturation, file_size_kb,
           histogram
    FROM images ORDER BY id DESC
''')


        images = cursor.fetchall()

    return render_template('index.html', images=images,
                           annotation_label=annotation_label,
                           image_url=image_url)





# ==== Voir les histogrammes dans le tableau ====

@app.route('/histogram/<int:image_id>/<string:type>')
def histogram_image(image_id, type):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT path, histogram FROM images WHERE id=?", (image_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Image not found", 404

    image_path, histogram_json = row
    fig, ax = plt.subplots()

    if type == 'color':
        try:
            histogram = json.loads(histogram_json)
        except Exception as e:
            return f"Error parsing histogram data: {e}", 500

        # R, G, B sont dans histogram (16 bins chacun)
        bins = list(range(16))
        ax.plot(bins, histogram[0:16], color='red', label='R')
        ax.plot(bins, histogram[16:32], color='green', label='G')
        ax.plot(bins, histogram[32:48], color='blue', label='B')
        ax.set_title("Histogramme couleur (16 bins)")
        ax.legend()

    elif type == 'luminance':
        # Recharger l'image et calculer histogramme luminance
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Image file not found", 404
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_title("Histogramme luminance (256 bins)")

    else:
        return "Invalid histogram type", 400

    ax.set_xlabel("Intensité")
    ax.set_ylabel("Nombre de pixels")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


# ==== Contours tableau ====
@app.route('/contour/<int:image_id>')
def contour_image(image_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT path FROM images WHERE id=?", (image_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Image non trouvée", 404

    image_path = row[0]
    filename = os.path.basename(image_path)
    contour_dir = os.path.join("static", "contour")
    os.makedirs(contour_dir, exist_ok=True)  # Crée le dossier s’il n’existe pas
    contour_path = os.path.join(contour_dir, filename)

    # Générer l'image avec contours si elle n'existe pas
    if not os.path.exists(contour_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Fichier image introuvable", 404
        edges = cv2.Canny(img, 100, 200)
        cv2.imwrite(contour_path, edges)

    return send_file(contour_path, mimetype='image/png')




# ====================== Routes pour annotation à la main + supression ======================

@app.route('/annotate/<int:image_id>/<label>')
def annotate(image_id, label):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE images SET annotation = ? WHERE id = ?', (label, image_id))
        conn.commit()
    return redirect(url_for('index'))

@app.route('/annotate/<int:image_id>', methods=['POST'])
def annotate_image(image_id):
    annotation = request.form.get('annotation')
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE images SET annotation = ? WHERE id = ?', (annotation, image_id))
        conn.commit()
    return redirect(url_for('index'))

@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        if result:
            filepath = result[0]
            
            # Supprimer fichier principal (le crop ou l'original enregistré)
            if os.path.exists(filepath):
                os.remove(filepath)

            # Nom de base pour YOLO
            detected_name = os.path.basename(filepath)

            # Supprimer l'image brute 
            upload_dir = os.path.join("static", "uploads")
            original_file = os.path.join(upload_dir, detected_name)
            if os.path.exists(original_file):
                os.remove(original_file)

            # Supprimer image annotée YOLO
            detected_dir = os.path.join("static", "uploads", "detect_poubelle")
            detected_file = os.path.join(detected_dir, detected_name)
            if os.path.exists(detected_file):
                os.remove(detected_file)

            # Supprimer crop YOLO
            crop_dir = os.path.join("static", "uploads", "detect_poubelle", "crops", "poubelle")
            crop_file = os.path.join(crop_dir, detected_name)
            if os.path.exists(crop_file):
                os.remove(crop_file)

        # Supprimer ligne BDD
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()

    return redirect(url_for('index'))


# ====================================================================================

@app.route('/static/uploads/<path:filepath>')
def uploaded_file(filepath):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    # Vérifie que le fichier existe
    if not os.path.exists(full_path):
        return "Fichier non trouvé", 404
    # Retourne le fichier
    return send_file(full_path)


# ====================== Routes naviguation images accueil ======================


@app.route('/imageview')
def imageview_index():
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images ORDER BY id ASC LIMIT 1').fetchone()
    conn.close()
    if image:
        return redirect(url_for('show_image', image_id=image['id']))
    return "Aucune image disponible."


@app.route('/image/<int:image_id>')
def show_image(image_id):
    conn = get_db_connection()
    image = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
    if not image:
        return redirect(url_for('imageview_index'))

    # Image précédente (circulaire)
    prev_img = conn.execute('SELECT id FROM images WHERE id < ? ORDER BY id DESC LIMIT 1', (image_id,)).fetchone()
    if not prev_img:
        prev_img = conn.execute('SELECT id FROM images ORDER BY id DESC LIMIT 1').fetchone()  # la dernière

    # Image suivante (circulaire)
    next_img = conn.execute('SELECT id FROM images WHERE id > ? ORDER BY id ASC LIMIT 1', (image_id,)).fetchone()
    if not next_img:
        next_img = conn.execute('SELECT id FROM images ORDER BY id ASC LIMIT 1').fetchone()  # la première

    conn.close()

    prev_id = prev_img['id'] if prev_img else None
    next_id = next_img['id'] if next_img else None

    return render_template('image.html', image=image, prev_id=prev_id, next_id=next_id)





# ====================== Routes Dashboard  ======================

@app.route("/stats")
def stats_page():
    return render_template("stats.html")


@app.route("/api/stats")
def api_stats():
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # Nombre total d'images
    cursor.execute("SELECT COUNT(*) FROM images")
    total_images = cursor.fetchone()[0]


    # Répartition pleine / vide
    cursor.execute("SELECT annotation, COUNT(*) FROM images GROUP BY annotation")
    annotations = dict(cursor.fetchall())

    # Histogramme des tailles de fichiers (en Ko)
    cursor.execute("SELECT file_size_kb FROM images")
    sizes = [f[0] for f in cursor.fetchall()]

    bin_edges = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 2000 ]
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
    bins = {label: 0 for label in bin_labels}
    for size in sizes:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= size < bin_edges[i+1]:
                bins[bin_labels[i]] += 1
                break

    file_size_histogram = bins

    # Contrastes
    cursor.execute("SELECT edge_density FROM images")
    contrastes = [round(c[0], 4) for c in cursor.fetchall() if c[0] is not None]

    # Répartition par date
    cursor.execute("SELECT upload_date, annotation FROM images")
    date_data = {}
    for date, ann in cursor.fetchall():
        if date:
            date = date.split("T")[0]  # Normaliser à AAAA-MM-JJ uniquement
            if date not in date_data:
                date_data[date] = {"pleine": 0, "vide": 0}
            if ann in date_data[date]:
                date_data[date][ann] += 1

    # Répartition par "ville"
    cursor.execute("SELECT latitude, longitude FROM images")
    villes = {"Zone 1": 0, "Zone 2": 0, "Zone 3": 0}
    for lat, lon in cursor.fetchall():
        if lat and lon:
            if lat > 48.85:
                villes["Zone 1"] += 1
            elif lat > 48.80:
                villes["Zone 2"] += 1
            else:
                villes["Zone 3"] += 1

    # Images pour la carte

    cursor.execute("SELECT latitude, longitude, annotation, upload_date FROM images")
    images_geo = []

    for lat, lon, ann, date in cursor.fetchall():
        # Si coordonnées manquantes, générer des coordonnées fictives en Amérique du Sud
        if lat is None or lon is None:
            lat = round(random.uniform(-34.0, 5.0), 6)
            lon = round(random.uniform(-75.0, -35.0), 6)

        images_geo.append({
            "latitude": lat,
            "longitude": lon,
            "annotation": ann,
            "upload_date": date
    })
        

    # Répartition par pays pour le camembert
    geolocator = Nominatim(user_agent="trashinator")

    cursor.execute("SELECT latitude, longitude FROM images WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
    coords = cursor.fetchall()

    countries = {}

    for lat, lon in coords:
        try:
            location = geolocator.reverse((lat, lon), language='en', timeout=10)
            if location and 'country' in location.raw['address']:
                country = location.raw['address']['country']
                countries[country] = countries.get(country, 0) + 1
            time.sleep(1)  # pour éviter de se faire bloquer par Nominatim
        except Exception as e:
            continue  # ignorer les erreurs

    conn.close()

    return jsonify({
    "total_images": total_images,
    "annotations": annotations,
    "file_size_histogram": file_size_histogram,
    "contrastes": contrastes,
    "date_distribution": date_data,
    "par_ville": villes,
    "par_pays": countries,
    "locations": images_geo  
})



# ====================== Route pour la carte GPS dans le dashboard ======================


@app.route("/api/gps_coordinates")
def gps_coordinates():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT filename, annotation, upload_date, latitude, longitude FROM images")
        rows = cursor.fetchall()

    results = []
    for row in rows:
        lat, lon = row["latitude"], row["longitude"]

        # Si les coordonnées sont manquantes, on en génère aléatoirement en Amérique du Sud
        if lat is None or lon is None:
            lat = round(random.uniform(-35.0, 5.0), 6)    # Latitude sud (ex: Brésil, Argentine)
            lon = round(random.uniform(-75.0, -35.0), 6)  # Longitude ouest

        results.append({
            "filename": row["filename"],
            "annotation": row["annotation"],
            "upload_date": row["upload_date"],
            "latitude": lat,
            "longitude": lon
        })

    return jsonify(results)

# ===========================================================================


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
