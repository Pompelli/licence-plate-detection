import cv2
import numpy as np
import imutils
import os

# Function to read the label file and parse all bounding boxes
def read_label(file_path):
    """
    Liest die Label-Datei und extrahiert alle Bounding-Boxen für ein Bild.
    Jede Zeile enthält das Label und die Bounding-Box-Koordinaten.
    """
    with open(file_path, 'r') as f:
        data = f.read().splitlines()  # Liest alle Zeilen in der Label-Datei
        labels_and_boxes = []
        
        # Iteriere durch alle Zeilen und extrahiere die Label- und Bounding-Box-Werte
        for line in data:
            components = line.split()
            label = int(components[0])  # Der erste Wert ist das Label (Klasse)
            # Extrahiere die nächsten vier Werte (x_center, y_center, width, height)
            x_center, y_center, width, height = map(float, components[1:])
            labels_and_boxes.append((label, x_center, y_center, width, height))  # Füge die Box zur Liste hinzu
    
    return labels_and_boxes  # Gibt eine Liste der Bounding-Boxen zurück

# Function to compare bounding boxes (from the image and label)
def compare_bounding_boxes(image_bbox, label_bbox, image_shape):
    """
    Vergleicht die Bounding-Box des Bildes (aus den erkannten Konturen) mit den
    Bounding-Box-Koordinaten aus der Label-Datei und überprüft, ob eine Übereinstimmung
    vorliegt.
    """
    # Extrahiere die Bounding-Box-Koordinaten des Bildes (aus den Konturen)
    x, y, w, h = image_bbox
    
    # Extrahiere Höhe und Breite des Bildes (ignoriere die Farbkanäle)
    height, width = image_shape[:2]  # Nur Höhe und Breite verwenden, Kanäle ignorieren
    
    # Konvertiere die normalisierten Label-Bounding-Box-Koordinaten in Pixelwerte
    label_x_center = int(label_bbox[1] * width)
    label_y_center = int(label_bbox[2] * height)
    label_width = int(label_bbox[3] * width)
    label_height = int(label_bbox[4] * height)
    
    # Berechne die Ecken der Label-Bounding-Box
    label_x1 = label_x_center - label_width // 2
    label_y1 = label_y_center - label_height // 2
    label_x2 = label_x_center + label_width // 2
    label_y2 = label_y_center + label_height // 2

    # Überprüfe, ob die Bild-Bounding-Box mit der Label-Bounding-Box überlappt
    return (label_x1 <= x + w <= label_x2 or label_x1 <= x <= label_x2) and \
           (label_y1 <= y + h <= label_y2 or label_y1 <= y <= label_y2)

# Haupt-Skript zur Verarbeitung aller Bilder im "pic"-Ordner
pic_folder = "Example_pics/pic"  # Ordner, in dem sich die Bilder befinden
label_folder = "Example_pics/lable"  # Ordner, in dem sich die Label-Dateien befinden

# Zähler für richtige Übereinstimmungen
correct_count = 0
# Zähler für die Anzahl der verarbeiteten Bilder
total_count = 0

# Durchlaufe alle Dateien im "pic"-Ordner
for image_name in os.listdir(pic_folder):
    if image_name.endswith(('.jpg', '.png')):  # Überprüfe, ob es sich um eine Bilddatei handelt
        total_count += 1  # Erhöhe die Gesamtanzahl der Bilder

        # Lade das Bild
        img_path = os.path.join(pic_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Image {image_name} not found. Skipping...")
            continue
        
        # Konvertiere das Bild in Graustufen
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Wende eine bilaterale Filterung an, um Rauschen zu reduzieren
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

        # Führe eine Kantenerkennung durch (Canny-Edge-Detection)
        edged = cv2.Canny(bfilter, 50, 150)


        # Finde die Konturen im Bild
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Durchlaufe alle Konturen und versuche, eine mit 4 Punkten (Rechteck) zu finden
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Wenn die Kontur 4 Ecken hat (ein Rechteck)
                location = approx
                break

        if location is None:
            print(f"Error: No contour with 4 vertices found in {image_name}. Skipping...")
            continue

        # Erstelle eine Maske für das erkannte Rechteck
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], -1, 255, -1)

        # Hole die Bounding-Box der erkannten Kontur
        x, y, w, h = cv2.boundingRect(location)
        image_bbox = (x, y, w, h)  # Diese Bounding-Box ist die Erkennung des Bildes

        # Lade die zugehörige Label-Datei
        label_file = os.path.join(label_folder, image_name.split('.')[0] + '.txt')
        if not os.path.exists(label_file):
            print(f"Label file for {image_name} not found. Skipping...")
            continue

        # Lese alle Bounding-Boxen aus der Label-Datei
        labels_and_boxes = read_label(label_file)

        # Überprüfe, ob eine der Label-Bounding-Boxen mit der erkannten Bounding-Box übereinstimmt
        match_found = False
        for _, x_center, y_center, width, height in labels_and_boxes:
            if compare_bounding_boxes(image_bbox, (_, x_center, y_center, width, height), img.shape):
                match_found = True
                break  # Wenn eine Übereinstimmung gefunden wurde, breche die Schleife ab

        if match_found:
            print(f"Match found for {image_name}")  # Wenn eine Übereinstimmung gefunden wurde
            correct_count += 1  # Erhöhe den Zähler für korrekte Übereinstimmungen
        else:
            print(f"No match for {image_name}")  # Wenn keine Übereinstimmung gefunden wurde

# Gib die endgültigen Ergebnisse aus
print(f"Total images processed: {total_count}")
print(f"Correct matches: {correct_count}")
print(f"Accuracy: {correct_count / total_count * 100:.2f}%")  # Berechne die Genauigkeit in Prozent
