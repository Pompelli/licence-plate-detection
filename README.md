# licence-plate-detection
licence plate detection

Dataset-source https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset?resource=download
Ich habe nur die Test Daten genommen. es gibt VIEL mehr wenn man will.



YOLO-Annotationsdaten für die Positionen der Objekte in einem Bild. 

Die YOLO-Annotationen bestehen aus fünf Werten pro Objekt, getrennt durch Leerzeichen:

Erste Zahl: Klassen-ID  Diese ID steht für eine bestimmte Objektklasse, die das Modell erkennen soll. Hier bedeutet 0, dass es sich um die Klasse "Kennzeichen" handelt.

Zweite Zahl: Die x-Koordinate des Mittelpunkts der Bounding Box (normalisiert). Hier ist das 0.234472875. Der Wert liegt zwischen 0 und 1, da YOLO die Positionen relativ zur Bildgröße angibt. Ein Wert von 0 wäre ganz links im Bild, und 1 wäre ganz rechts.

Dritte Zahl: Die y-Koordinate des Mittelpunkts der Bounding Box (ebenfalls normalisiert). Hier ist das 0.5047654166666667. Ein Wert von 0 wäre ganz oben im Bild, und 1 wäre ganz unten.

Vierte Zahl: Die Breite der Bounding Box (ebenfalls normalisiert). Hier ist das 0.028085. Diese Zahl gibt die Breite der Box relativ zur Bildbreite an.

Fünfte Zahl: Die Höhe der Bounding Box (normalisiert). Hier ist es 0.014829999999999973. Diese Zahl gibt die Höhe der Box relativ zur Bildhöhe an.

Beispiel für die Interpretation der Zeilen:
Zeile 1:
0: Klassen-ID (Kennzeichen).
0.234472875: x-Koordinate des Box-Mittelpunkts, etwa 23 % vom linken Bildrand entfernt.
0.5047654166666667: y-Koordinate des Box-Mittelpunkts, etwa in der Bildmitte.
0.028085: Breite der Box, ca. 2,8 % der gesamten Bildbreite.
0.014829999999999973: Höhe der Box, ca. 1,5 % der gesamten Bildhöhe.
Zeile 2:
0: Klassen-ID (Kennzeichen).
0.67940375: x-Koordinate des Box-Mittelpunkts, etwa 68 % vom linken Bildrand entfernt.
0.7933044166666666: y-Koordinate des Box-Mittelpunkts, relativ weit unten.
0.11842399999999997: Breite der Box, ca. 11,8 % der Bildbreite.
0.075703: Höhe der Box, ca. 7,6 % der Bildhöhe.
Zusammengefasst beschreiben diese Werte die Position und Größe der Rechtecke (Bounding Boxes), die die Objekte im Bild umgeben.