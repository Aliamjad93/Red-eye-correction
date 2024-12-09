import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import cv2
import numpy as np

def detect_and_correct_red_iris_gpu(image, eyes_boxes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_image = image.copy()
    
    for eye_box in eyes_boxes:
        x, y, w, h = eye_box
        eye_image = image[y:y+h, x:x+w]

        eye_tensor = torch.tensor(eye_image.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0
        b, g, r = eye_tensor[0], eye_tensor[1], eye_tensor[2]

        bg = b + g
        
        mask = ((r > (bg - 0.04)) & (r > 0.4)) & (r>0.5).type(torch.uint8)

        mask_cpu = (mask * 255).byte().cpu().numpy()

        contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            mask_cpu = np.zeros_like(mask_cpu)
            cv2.drawContours(mask_cpu, [max_contour], 0, (255), -1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_cpu = cv2.morphologyEx(mask_cpu, cv2.MORPH_CLOSE, kernel)
            mask_cpu = cv2.dilate(mask_cpu, kernel, iterations=3)

            mask = torch.tensor(mask_cpu, device=device, dtype=torch.float32) / 255.0

            mean = (bg / 2).clamp(0, 1)

            masked_mean = mean * mask

            eye_tensor[2] = eye_tensor[2] * (1 - mask) + masked_mean

            eye_corrected = (eye_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)

            output_image[y:y+h, x:x+w] = eye_corrected

    return output_image

# image = cv2.imread(r"C:\Users\farha\Downloads\WhatsApp Image 2024-11-17 at 2.25.13 AM.jpeg")
image=cv2.imread(r"C:\Users\farha\OneDrive\Desktop\images (2).jpg")
kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
sharpened_image = cv2.filter2D(image, -1, kernel)

gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=30, minSize=(30, 30))


if len(eyes) > 0:
    print('Eyes is detected')
    corrected_image = detect_and_correct_red_iris_gpu(image, eyes)
    corrected_image=cv2.resize(corrected_image,(900,1000))
    cv2.imwrite(r'C:\Users\farha\Downloads\new2.jpg', corrected_image)
    cv2.imshow(r'new',corrected_image)
    cv2.waitKey()

# eyess=[[72, 82, 27, 48], [123, 66, 42, 42]]

# corrected_image = detect_and_correct_red_iris_gpu(image, eyess)
# corrected_image=cv2.resize(corrected_image,(900,1000))
# cv2.imwrite(r'C:\Users\farha\Downloads\new2.jpg', corrected_image)
# cv2.imshow(r'new',corrected_image)
# cv2.waitKey()




