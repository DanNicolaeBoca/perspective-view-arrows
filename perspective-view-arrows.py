import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
#import os

white = (255, 255, 255)
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
purple = (255, 0, 255)
cyan = (255, 255, 0)

# Initialize points and state variables
points = []
dragging = False
current_point_index = -1

# Load points from file if it exists
try:
    with open("points.json", "r") as file:
        points = json.load(file)
except FileNotFoundError:
    points = []


# Function to draw points and lines on a copy of the image
def draw_points(img_copy):
    for point in points:
        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
    if len(points) == 4:
        for i in range(4):
            cv2.line(img_copy, points[i], points[(i + 1) % 4], (255, 0, 0), 2)
    return img_copy


# Function to apply perspective transformation and show result
def update_perspective():
    if len(points) == 4:
        width, height = 500, 700  # Adjust the size based on your needs
        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        src_pts = np.array(points, dtype="float32")
        H, _ = cv2.findHomography(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(img, H, (width, height))
        cv2.imshow("Comparison", warped_img)


# Mouse callback function
def select_points(event, x, y, flags, param):
    global points, dragging, current_point_index
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(points):
            if abs(point[0] - x) < 5 and abs(point[1] - y) < 5:
                dragging = True
                current_point_index = i
                break
        else:
            if len(points) < 4:
                points.append((x, y))
        img_copy = img.copy()
        img_copy = draw_points(img_copy)
        cv2.imshow("Image", img_copy)
        update_perspective()
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        points[current_point_index] = (x, y)
        img_copy = img.copy()
        img_copy = draw_points(img_copy)
        cv2.imshow("Image", img_copy)
        update_perspective()
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        current_point_index = -1
        img_copy = img.copy()
        img_copy = draw_points(img_copy)
        cv2.imshow("Image", img_copy)
        update_perspective()


def save_trackbars_values(
    startPointX, startPointY, slotSize, imageTemplateFetcher, slotNumbers, lineThickness
):
    values = {
        "startPointX": startPointX,
        "startPointY": startPointY,
        "slotSize": slotSize,
        "imageTemplateFetcher": imageTemplateFetcher,
        "slotNumbers": slotNumbers,
        "lineThickness": lineThickness,
    }
    with open("trackbar_values.json", "w") as file:
        json.dump(values, file)


def extractImageFromGridTemplate(
    img, slotSize, templateSize, startingPointX, startingPointY
):
    startingPointX = int(startingPointX - (templateSize - slotSize) / 2)
    startingPointY = int(startingPointY - (templateSize + slotSize) / 2)
    endPointX = startingPointX + templateSize
    endPointY = startingPointY + templateSize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[startingPointY + 1 : endPointY, startingPointX + 1 : endPointX]


def extractImageFromGrid(img, slotSize, startingPointX, startingPointY):
    endPointX = startingPointX + slotSize
    endPointY = startingPointY + slotSize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[startingPointY:endPointY, startingPointX:endPointX]


def fillWithImagesTemplate(
    img, slotSize, slotNumbers, percentSlotSize, startPoint_X, startPoint_Y
):
    # Atribuirea dimensiunilor șablonului căutat
    num_slots_x = slotNumbers
    num_slots_y = slotNumbers

    templateSize = int(slotSize * (percentSlotSize / 100))

    # Inițializarea variabilei searchedTemplate ca un tablou numpy gol.
    searchedTemplate = np.empty((num_slots_y, num_slots_x), dtype=object)

    # Popularea tabloului cu imagini extrase
    for i in range(0, num_slots_y, 1):
        for j in range(0, num_slots_x, 1):
            startX = startPoint_X + j * slotSize
            startY = startPoint_Y + (i + 1) * slotSize

            # Asigurarea faptului că punctele de start de unde se extrag imagini există
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0

            extracted_image = extractImageFromGridTemplate(
                img, slotSize, templateSize, startX, startY
            )
            searchedTemplate[i, j] = extracted_image

    return searchedTemplate


def fillWithImagesBigGrid(img, slotSize, slotNumbers, startPoint_X, startPoint_Y):
    # Atribuirea dimensiunilor șablonului căutat
    num_slots_x = slotNumbers
    num_slots_y = slotNumbers
    # Inițializarea variabilei searchedTemplate ca un tablou numpy gol.
    searchedTemplate = np.empty((num_slots_y, num_slots_x), dtype=object)

    # Popularea tabloului cu imagini extrase
    for i in range(0, num_slots_y, 1):
        for j in range(0, num_slots_x, 1):
            startX = startPoint_X + j * slotSize
            startY = startPoint_Y + i * slotSize

            # Asigurarea faptului că punctele de start de unde se extrag imagini există
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0

            extracted_image = extractImageFromGrid(img, slotSize, startX, startY)
            searchedTemplate[i, j] = extracted_image
            print(
                f"Filled searchedTemplate[{i},{j}] with shape {extracted_image.shape}"
            )  # Debug statement

    return searchedTemplate


def generateGrid(
    img,
    slotSize,
    slotNumbers,
    startingPointX,
    startingPointY,
    selectedColor,
    lineThickness,
):
    for i in range(0, slotNumbers + 1, 1):
        cv2.line(
            img,
            (startingPointX + i * slotSize, startingPointY),
            (startingPointX + i * slotSize, startingPointY + slotSize * slotNumbers),
            selectedColor,
            lineThickness,
        )
        cv2.line(
            img,
            (startingPointX, startingPointY + i * slotSize),
            (startingPointX + slotSize * slotNumbers, startingPointY + i * slotSize),
            selectedColor,
            lineThickness,
        )
        cv2.circle(img, (startingPointX, startingPointY), lineThickness + 4, red, -1)


def getArrowOrientation(array):
    print(f"Number of images: {np.size(array)}")
    for i, src in enumerate(array.flatten()):
        if src is None or not isinstance(src, np.ndarray):
            print(f"Image {i + 1} not found or invalid!")
            continue

        dst = np.zeros_like(src)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 177, 200, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            print(f"No contours found in image {i + 1}!")
            continue

        contour_points = np.vstack(contours).squeeze()
        num_points = len(contour_points)
        print(f"Image {i + 1} - Number of contour points: {num_points}")

        cv2.drawContours(dst, contours, -1, (255, 255, 255), 1, cv2.LINE_8, hierarchy)
        for point in contour_points:
            cv2.circle(dst, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)

        if len(contour_points) > 0:
            height, width = dst.shape[:2]
            top_half = contour_points[contour_points[:, 1] < height // 2]
            bottom_half = contour_points[contour_points[:, 1] >= height // 2]
            left_half = contour_points[contour_points[:, 0] < width // 2]
            right_half = contour_points[contour_points[:, 0] >= width // 2]

            top_count = len(top_half)
            bottom_count = len(bottom_half)
            left_count = len(left_half)
            right_count = len(right_half)

            print(
                f"Image {i + 1} - Top points: {top_count}, Bottom points: {bottom_count}, Left points: {left_count},"
                f" Right points: {right_count}"
            )

            if (
                top_count >= bottom_count
                and top_count >= left_count
                and top_count >= right_count
            ):
                orientation = "UpArrow"
            elif (
                bottom_count >= top_count
                and bottom_count >= left_count
                and bottom_count >= right_count
            ):
                orientation = "DownArrow"
            elif (
                left_count >= top_count
                and left_count >= bottom_count
                and left_count >= right_count
            ):
                orientation = "LeftArrow"
            else:
                orientation = "RightArrow"

            print(f"Image {i + 1} - Orientation: {orientation}")

            cv2.namedWindow(f"{i+1}", cv2.WINDOW_FREERATIO)
            cv2.resizeWindow(f"{i+1}", getImageSize_X(dst) * 8, getImageSize_Y(dst) * 8)
            cv2.imshow(f"{i+1}", dst)
            while True:
                key = cv2.waitKey(0) & 0xFF  # Wait for a key press
                if key == ord("n") and i < np.size(array) - 1:
                    break  # Proceed to the next image
                else:
                    # cv2.destroyAllWindows()
                    break


def getImageSize_X(image):
    return np.array(image, dtype=object).shape[1]


def getImageSize_Y(image):
    return np.array(image, dtype=object).shape[0]


def nothing(x):
    pass


def initTrackBars(image, windowTitle):
    # Creating trackbars
    cv2.createTrackbar("StartPoint_X", windowTitle, 0, getImageSize_X(image), nothing)
    cv2.createTrackbar("StartPoint_Y", windowTitle, 0, getImageSize_Y(image), nothing)
    cv2.createTrackbar("SlotSize", windowTitle, 2, getImageSize_X(image), nothing)
    cv2.createTrackbar("ImageTemplateFetcher", windowTitle, 10, 100, nothing)
    cv2.createTrackbar("SlotNumbers", windowTitle, 2, 10, nothing)
    cv2.createTrackbar("LineThickness", windowTitle, 1, 10, nothing)
    # Setting the default values for program start
    cv2.setTrackbarPos("StartPoint_X", windowTitle, 0)
    cv2.setTrackbarPos("StartPoint_Y", windowTitle, 0)
    cv2.setTrackbarPos("SlotSize", windowTitle, int(getImageSize_X(image) / 5))
    cv2.setTrackbarPos("ImageTemplateFetcher", windowTitle, 50)
    cv2.setTrackbarPos("SlotNumbers", windowTitle, 5)
    cv2.setTrackbarPos("LineThickness", windowTitle, 5)
    cv2.setTrackbarMin("StartPoint_X", windowTitle, 0)
    cv2.setTrackbarMin("StartPoint_Y", windowTitle, 0)
    cv2.setTrackbarMin("LineThickness", windowTitle, 1)


def load_trackbars_values():
    try:
        with open("trackbar_values.json", "r") as file:
            values = json.load(file)
            return (
                values["startPointX"],
                values["startPointY"],
                values["slotSize"],
                values["imageTemplateFetcher"],
                values["slotNumbers"],
                values["lineThickness"],
            )
    except FileNotFoundError:
        # Return default values if the file doesn't exist
        return 0, 0, 2, 50, 2, 5


def get_rectangle_points(center, width, height, angle):
    w, h = width / 2, height / 2
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
    corners = np.dot(corners, rotation_matrix[:, :2].T) + center
    return corners.astype(int)


# Function to draw rectangle
def draw_rectangle(img_copy):
    points = get_rectangle_points(rect_center, rect_width, rect_height, rect_angle)
    for i in range(4):
        cv2.line(img_copy, tuple(points[i]), tuple(points[(i + 1) % 4]), (255, 0, 0), 2)
    return img_copy


# Function to update the perspective view
def update_perspective_view():
    global warped_img
    width, height = 500, 500  # Adjust the size based on your needs
    dst_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    src_pts = get_rectangle_points(
        rect_center, rect_width, rect_height, rect_angle
    ).astype("float32")
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, H, (width, height))
    cv2.imshow("Perspective View", warped_img)


# Callback functions for trackbars
max_angle = 360.0  # Maximum angle in degrees
angle_factor = 10  # Increase factor for finer control


def update_angle(val):
    global rect_angle
    rect_angle = val / angle_factor  # Convert back to float angle
    img_copy = img.copy()
    img_copy = draw_rectangle(img_copy)
    cv2.imshow("Image", img_copy)
    update_perspective_view()


def update_width(val):
    global rect_width
    rect_width = val
    img_copy = img.copy()
    img_copy = draw_rectangle(img_copy)
    cv2.imshow("Image", img_copy)
    update_perspective_view()


def update_height(val):
    global rect_height
    rect_height = val
    img_copy = img.copy()
    img_copy = draw_rectangle(img_copy)
    cv2.imshow("Image", img_copy)
    update_perspective_view()


def update_center_x(val):
    global rect_center
    rect_center = (val, rect_center[1])
    img_copy = img.copy()
    img_copy = draw_rectangle(img_copy)
    cv2.imshow("Image", img_copy)
    update_perspective_view()


def update_center_y(val):
    global rect_center
    rect_center = (rect_center[0], val)
    img_copy = img.copy()
    img_copy = draw_rectangle(img_copy)
    cv2.imshow("Image", img_copy)
    update_perspective_view()


def crop_to_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    # Define a custom anti-aliasing kernel

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > image_area * 0.01]

    if len(contours) == 0:
        print("Error: No valid contours found.")
        return image  # No valid contours found after filtering

    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    x, y, w, h = cv2.boundingRect(contour)

    cropped_image = image[y : y + h, x : x + w]

    # some image processing
    n = 2
    k_num = n + 1
    d = 11
    sigma = 35

    for i in (1, n, 1):
        cropped_image = cv2.GaussianBlur(
            cropped_image, (2 * (k_num - i) + 1, 2 * (k_num - i) + 1), 0
        )
        cropped_image = cv2.bilateralFilter(
            cropped_image,
            d - i * 2 + 1,
            2 * (int(sigma / i)) + 1,
            2 * (int(sigma / i)) + 1,
        )
    return cropped_image


def crop_contours_in_images(searchedTemplate):
    rows, cols = searchedTemplate.shape
    croppedTemplate = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            croppedTemplate[i, j] = crop_to_contour(searchedTemplate[i, j])

    return croppedTemplate


# ===== Program Start ===== #
# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open file dialog
file_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=(
        ("JPEG files", "*.jpg;*.jpeg"),
        ("PNG files", "*.png"),
        ("All files", "*.*"),
    ),
)

# Check if a file was selected
if file_path:
    img = cv2.imread(file_path)
    if img is not None:
        # Process the image as needed
        print("Image loaded successfully!")
    else:
        print("Error loading image.")
else:
    print("No file selected.")


if img is None:
    print(f"Error: Could not load image from {image_path}.")
    exit()

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", select_points)

if points:
    img_copy = img.copy()
    img_copy = draw_points(img_copy)
    cv2.imshow("Image", img_copy)
    update_perspective()

while True:
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()

if len(points) == 4:
    # Save points to file for future use
    with open("points.json", "w") as file:
        json.dump(points, file)

    # Apply the final perspective transformation
    width, height = 500, 500  # Adjust the size based on your needs
    dst_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    src_pts = np.array(points, dtype="float32")
    H, _ = cv2.findHomography(src_pts, dst_pts)
    processed_img = cv2.warpPerspective(img, H, (width, height))

    windowName = "Perspective View Extraction"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, processed_img)
    initTrackBars(processed_img, windowName)
    (
        startPoint_X,
        startPoint_Y,
        slotSize,
        imageTemplateFetcher,
        slotNumbers,
        lineThickness,
    ) = load_trackbars_values()

frame_captured = False
# Loop to keep the processed image window open until 'q' is pressed
while True:
    if not frame_captured:
        img_copy = processed_img.copy()
        generateGrid(
            img_copy,
            slotSize,
            slotNumbers,
            startPoint_X,
            startPoint_Y,
            green,
            lineThickness,
        )

        # Setting the template grid size as percentage of the main grid slot size
        templateSize = int(slotSize * (imageTemplateFetcher / 100))

        for i in range(1, slotNumbers + 1, 1):
            for j in range(0, slotNumbers, 1):
                generateGrid(
                    img_copy,
                    templateSize,
                    1,
                    int(startPoint_X - (templateSize - slotSize) / 2) + j * slotSize,
                    int(startPoint_Y - (templateSize + slotSize) / 2) + i * slotSize,
                    yellow,
                    lineThickness,
                )

        cv2.imshow(windowName, img_copy)
        # Get current positions of all trackbars

        startPoint_X = cv2.getTrackbarPos("StartPoint_X", windowName)
        startPoint_Y = cv2.getTrackbarPos("StartPoint_Y", windowName)
        slotSize = cv2.getTrackbarPos("SlotSize", windowName)
        imageTemplateFetcher = cv2.getTrackbarPos("ImageTemplateFetcher", windowName)
        slotNumbers = cv2.getTrackbarPos("SlotNumbers", windowName)
        lineThickness = cv2.getTrackbarPos("LineThickness", windowName)
        # Save trackbar values
        save_trackbars_values(
            startPoint_X,
            startPoint_Y,
            slotSize,
            imageTemplateFetcher,
            slotNumbers,
            lineThickness,
        )
        cv2.setTrackbarMin("StartPoint_X", windowName, 0)
        cv2.setTrackbarMin("StartPoint_Y", windowName, 0)
        cv2.setTrackbarMin("ImageTemplateFetcher", windowName, 10)
        cv2.setTrackbarMax(
            "StartPoint_X",
            windowName,
            getImageSize_X(img_copy) - slotNumbers * slotSize,
        )
        cv2.setTrackbarMax(
            "StartPoint_Y",
            windowName,
            getImageSize_Y(img_copy) - slotNumbers * slotSize,
        )

        if cv2.waitKey(1) & 0xFF == ord("c"):
            frame_captured = True
            processed_img = cv2.bilateralFilter(processed_img, 15, 75, 75)

            searchedTemplate = fillWithImagesTemplate(
                processed_img,
                slotSize,
                slotNumbers,
                imageTemplateFetcher,
                startPoint_X,
                startPoint_Y,
            )

            croppedTemplate = crop_contours_in_images(searchedTemplate)

            getArrowOrientation(croppedTemplate)
            plt.figure()  # Adjust the size as needed
            counter = 1
            resize_width = 100
            resize_height = 100
            for a in range(0, slotNumbers):
                for b in range(0, slotNumbers):
                    resized_image = cv2.resize(
                        croppedTemplate[a][b], (resize_width, resize_height)
                    )
                    ax = plt.subplot(slotNumbers, slotNumbers, counter)
                    plt.imshow(resized_image)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                    counter = counter + 1
    else:
        if cv2.waitKey(10) & 0xFF == ord("q"):
            plt.show()
            break
