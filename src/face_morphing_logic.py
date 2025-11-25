# src/face_morph_logic.py
import cv2 as cv
import numpy as np
import dlib
import imageio.v2 as imageio
import streamlit as st
import io

# --- Caricamento Modelli (Cache) (RIMANE INVARIATO) ---
@st.cache_resource
def get_dlib_predictor():
    # ATTENZIONE: Questo file DEVE essere in models/cv/
    predictor_path = "models/cv/shape_predictor_68_face_landmarks.dat"
    try:
        shape_predictor = dlib.shape_predictor(predictor_path)
        return shape_predictor
    except Exception as e:
        st.error(f"Errore nel caricamento di 'shape_predictor_68_face_landmarks.dat'. Assicurati che sia in 'models/cv/'. Errore: {e}")
        return None

@st.cache_resource
def get_face_cascade():
    cascade_path = "models/cv/haarcascade_frontalface_alt.xml" #
    face_cascade = cv.CascadeClassifier()
    if not face_cascade.load(cv.samples.findFile(cascade_path)):
        st.error(f"Errore nel caricamento di 'haarcascade_frontalface_alt.xml'. Assicurati che sia in 'models/cv/'.")
        return None
    return face_cascade

# --- Funzioni di base (da face_morphing.py - RIMANGONO INVARIATE) ---
# (align_faces, get_landmark_triangoli, calcolaLandmarkIntermedi, transformazione)

def align_faces(image1, image2, face_cascade):
    # ... (Il corpo della funzione align_faces rimane invariato) ...
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces1) > 1:
        faces1 = [max(faces1, key=lambda rect: rect[2] * rect[3])]
    if len(faces2) > 1:
        faces2 = [max(faces2, key=lambda rect: rect[2] * rect[3])]
    if len(faces1) == 0 or len(faces2) == 0:
        print("Errore: Non sono stati trovati volti in una o entrambe le immagini.")
        return None, None
    x1, y1, w1, h1 = faces1[0]
    x2, y2, w2, h2 = faces2[0]
    pts1 = np.float32([[x1, y1], [x1 + w1, y1], [x1, y1 + h1]])
    pts2 = np.float32([[x2, y2], [x2 + w2, y2], [x2, y2 + h2]])
    M = cv.getAffineTransform(pts2, pts1)
    immagine_allineata = cv.warpAffine(image2, M, (image2.shape[1], image2.shape[0]), borderMode=cv.BORDER_REPLICATE)
    
    # Ritorna anche l'immagine 1 originale per coerenza
    return image1, immagine_allineata


def get_landmark_triangoli(img, shape_predictor, face_cascade):
    # ... (Il corpo della funzione get_landmark_triangoli rimane invariato) ...
    height, width = img.shape[:2]
    img_scala_grigi = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rettangoli = face_cascade.detectMultiScale(img_scala_grigi, scaleFactor=1.1 , minNeighbors=5, minSize=(100, 100))
    subdiv = cv.Subdiv2D((0, 0, width, height))
    landmarks_list=list() 
    if len(rettangoli) == 0:
        print("Nessun volto rilevato.")
        return None, None, None
    elif len(rettangoli) >= 1:
        rettangoli = [max(rettangoli, key=lambda r: r[2] * r[3])]
    x, y, w, h = rettangoli[0]
    rettangolo_predictor_dlib = dlib.rectangle(x, y, x + w, y + h)
    landmarks = shape_predictor(img_scala_grigi, rettangolo_predictor_dlib)
    landmarks_list = list(landmarks.parts())
    landmarks_list.append(dlib.point(0, 0))
    landmarks_list.append(dlib.point(0, height - 1))
    landmarks_list.append(dlib.point(width - 1, 0))
    landmarks_list.append(dlib.point(width - 1, height - 1))
    points = np.array([(pt.x, pt.y) for pt in landmarks_list],dtype=np.float32)
    subdiv.insert(points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    triangles_indici = []
    dizionario_punti_indici = {(pt.x, pt.y): i for i, pt in enumerate(landmarks_list)}
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(pt in dizionario_punti_indici for pt in pts):
            triangle_indices = [dizionario_punti_indici[pt] for pt in pts]
            triangles_indici.append(triangle_indices)
    return triangles_indici, landmarks_list


def calcolaLandmarkIntermedi(landmark_img_src, landmark_img_dst, numero_step):
    # ... (Il corpo della funzione calcolaLandmarkIntermedi rimane invariato) ...
    landmarks_intermedi=[]
    for i in range(numero_step + 1): # +1 per includere l'ultimo frame
        t = i / numero_step 
        src_points = np.array([(pt.x, pt.y) for pt in landmark_img_src])
        dst_points = np.array([(pt.x, pt.y) for pt in landmark_img_dst])
        landmark_intermedio = (1 - t) * src_points + t * dst_points
        landmarks_intermedi.append(landmark_intermedio)
    return landmarks_intermedi


def transformazione(immagine, landamark_img_src, landmark_intermedi_img, triangoli_indici):

    rows, cols = immagine.shape[:2]
    x_map_base, y_map_base = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    frame_intermedi = []
    for landamark_step_corrispondente in landmark_intermedi_img:
        x_map = x_map_base.copy()
        y_map = y_map_base.copy()
        for tri_ind in triangoli_indici:
            punti_triangolo_src = np.float32([[landamark_img_src[i].x, landamark_img_src[i].y] for i in tri_ind])
            punti_triangolo_dst = np.float32([landamark_step_corrispondente[i] for i in tri_ind])
            mask = np.zeros((rows, cols), dtype=np.uint8)
            cv.fillConvexPoly(mask,punti_triangolo_src.astype(np.int32),255)
            A = cv.getAffineTransform(punti_triangolo_src, punti_triangolo_dst)
            A_homogeneous = np.vstack([A, [0, 0, 1]])
            A_inv = np.linalg.inv(A_homogeneous)
            indici_riga_triangolo,indici_colonna_triangolo = np.where(mask == 255)
            coords = np.dot(A_inv, np.vstack([indici_colonna_triangolo,indici_riga_triangolo, np.ones(indici_riga_triangolo.shape[0])]))
            x_map[indici_riga_triangolo, indici_colonna_triangolo] = coords[0, :].astype(np.float32)
            y_map[indici_riga_triangolo, indici_colonna_triangolo] = coords[1, :].astype(np.float32)
        frame = cv.remap(immagine, x_map, y_map, interpolation=cv.INTER_LINEAR)
        frame_intermedi.append(frame)
    return frame_intermedi

# --- Logica per la SINGOLA transizione (ex generate_morph_gif) ---

def _generate_single_morph_frames(img1_cv, img2_cv, shape_predictor, face_cascade, numero_step=30):
    """
    Funzione ausiliaria per generare la lista di frame RGB (numpy arrays) 
    per la transizione tra due immagini CV.
    """
    # 1. Allinea Immagine 2 su Immagine 1
    img_src, img_dst = align_faces(img1_cv, img2_cv, face_cascade)
    if img_dst is None:
        raise Exception("Allineamento fallito. Assicurati che ci sia un volto chiaro in entrambe le immagini.")

    # 2. Trova Landmark e Triangoli
    triangoli_src, landmark_src = get_landmark_triangoli(img_src, shape_predictor, face_cascade)
    triangoli_dst, landmark_dst = get_landmark_triangoli(img_dst, shape_predictor, face_cascade)
    
    if landmark_src is None or landmark_dst is None:
        raise Exception("Calcolo dei landmark fallito.")

    # 3. Calcola Landmark Intermedi
    landmark_intermedi_src_dst = calcolaLandmarkIntermedi(landmark_src, landmark_dst, numero_step)
    landmark_intermedi_dst_src = calcolaLandmarkIntermedi(landmark_dst, landmark_src, numero_step)

    # 4. Warping (Trasformazione)
    frame_src_dst = transformazione(img_src, landmark_src, landmark_intermedi_src_dst, triangoli_src)
    frame_dst_src = transformazione(img_dst, landmark_dst, landmark_intermedi_dst_src, triangoli_dst)

    # 5. Blending
    frames_rgb = []
    for k in range(numero_step + 1):
        t = k / numero_step
        # Blending (cross-dissolve)
        blended_frame = cv.addWeighted(frame_src_dst[k], 1 - t, frame_dst_src[numero_step - k], t, 0)
        # Per GIF (imageio vuole RGB)
        blended_frame_rgb = cv.cvtColor(blended_frame, cv.COLOR_BGR2RGB)
        frames_rgb.append(blended_frame_rgb)
        
    return frames_rgb

# --- Funzione Principale per il Morphing a Catena (NUOVA) ---

def generate_chained_morph_gif(image_pil_list, numero_step=30, resize_factor=0.6):
    """
    Orchestra il morphing a catena tra una lista di immagini PIL e ritorna un GIF in bytes.
    """
    if len(image_pil_list) < 2:
        raise ValueError("Sono richieste almeno due immagini per il morphing a catena.")

    shape_predictor = get_dlib_predictor()
    face_cascade = get_face_cascade()
    
    if shape_predictor is None or face_cascade is None:
        raise Exception("Modelli CV (dlib o opencv) non caricati.")

    all_frames_rgb = []
    
    # 1. Uniformare e convertire tutte le immagini in OpenCV e ridimensionarle
    # Usiamo la prima immagine come riferimento per le dimensioni finali
    img_ref = image_pil_list[0]
    width = int(img_ref.width * resize_factor)
    height = int(img_ref.height * resize_factor)
    
    cv_images = []
    for img_pil in image_pil_list:
        # Converti PIL in OpenCV (RGB -> BGR)
        img_cv = cv.cvtColor(np.array(img_pil.convert('RGB')), cv.COLOR_RGB2BGR)
        # Resize all'uniformità
        img_cv = cv.resize(img_cv, (width, height))
        cv_images.append(img_cv)

    # 2. Processa la catena (i -> i+1)
    for i in range(len(cv_images) - 1):
        img_src_cv = cv_images[i]
        img_dst_cv = cv_images[i+1]
        
        # Aggiungo un feedback per l'utente su Streamlit
        st.info(f"Elaborazione transizione: Immagine {i+1} -> Immagine {i+2}...")

        frames_transition = _generate_single_morph_frames(
            img_src_cv, 
            img_dst_cv, 
            shape_predictor, 
            face_cascade, 
            numero_step
        )
        
        # Aggiungiamo tutti i frame, ma escludiamo l'ultimo (il frame di destinazione)
        # se non è l'ultima transizione, per evitare di duplicare il frame 
        # tra la fine di una transizione e l'inizio della successiva.
        if i < len(cv_images) - 2:
            # Rimuovi l'ultimo frame (img i+1) per evitare ripetizioni
            all_frames_rgb.extend(frames_transition[:-1])
        else:
            # Se è l'ultima transizione, aggiungi tutti i frame, incluso l'ultimo
            all_frames_rgb.extend(frames_transition)
    
    if not all_frames_rgb:
        raise Exception("Nessun frame generato. Controlla che le immagini abbiano volti validi.")

    # 3. Salva GIF in memoria
    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, all_frames_rgb, format='GIF', duration=0.1) # Durata: 0.1s per frame
    
    return gif_buffer.getvalue()

