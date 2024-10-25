import cv2
import numpy as np

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, image_list):
        # Converting images from RGB to BGR
        image_list_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in image_list]

        # Detect and extract features using SIFT algo
        keypoints, descriptors = self.detect_and_extract_features(image_list_bgr)
        # keypoints, descriptors = self.detect_and_extract_features(image_list)

        # Match features between images using FLANN-based matcher
        matches = self.match_features(descriptors)

        # Estimate homography matrices
        homographies = self.estimate_homographies(matches, keypoints)

        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(image_list_bgr)
        if status == cv2.Stitcher_OK:
            #for consistent display
            stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
            return stitched_image_rgb, homographies
        else:
            print("Error: Unable to stitch images.")
            return None, homographies


        # stitched_image = self.stitch_images(image_list, homographies)

        # if stitched_image is not None:
        #     return stitched_image, homographies
        # else:
        #     print("Error: Unable to stitch images.")
        #     return None, homographies

    def stitch_images(self, image_list, homographies):
        print("entered stich images")
        #initial canvas size based on input image size
        h, w, _ = image_list[0].shape
        canvas = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)  # should be large enough to hold the panorama

        # Compute the inverse transformations for placing images on the canvas
        current_homography = np.eye(3)
        canvas_center = (canvas.shape[1] // 2, canvas.shape[0] // 2)

        for i, image in enumerate(image_list):
            # For the first image  no homography needed
            if i == 0:
                warped_image = image
            else:
                # Multiply current_homography by each homography in sequence
                current_homography = current_homography @ homographies[i - 1]

                warped_image = self.apply_homography(image, current_homography, canvas.shape[:2])

            # Blend the warped image onto the canvas
            self.blend_images(canvas, warped_image, canvas_center)

        return canvas
    

    def match_features(self, descriptors):
        #Fast Library for Approximate Nearest Neighbors
        index_params = dict(algorithm=1, trees=5)#using KD-Tree
        search_params = dict(checks=50)#no. of recursive check leaf nodes
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        matches = []
        for i in range(len(descriptors) - 1):
            if descriptors[i] is not None and descriptors[i + 1] is not None:
                #find two best matches for each pair of descriptors
                match = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)

                """ Apply Lowe's ratio test to retain only those matches wher
                distance of the best match is less than 75% of the second-best"""
                good_matches = [m for m, n in match if m.distance < 0.75 * n.distance]

                if len(good_matches) > 10:
                    matches.append(good_matches)
                else:
                    print(f"Not enough matches between images {i} and {i + 1}. Skipping.")
        return matches

    def detect_and_extract_features(self, image_list):
        sift = cv2.SIFT_create()
        keypoints = []
        descriptors = []
        for img in image_list:
            kp, desc = sift.detectAndCompute(img, None)#detect keypoints and descriptors
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors
    
    def estimate_homographies(self, matches, keypoints):
        homographies = []
        for i, match_set in enumerate(matches):
            if len(match_set) < 4:
                print(f"Not enough matches to compute homography for image pair {i}. Skipping.")
                continue

            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 2)#from matched keypoints
            dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 2)

            # Manually compute the homography using Direct Linear Transform (DLT)
            H = self.compute_homography(src_pts, dst_pts)
            if H is not None:
                homographies.append(H)

        return homographies

    def compute_homography(self, src_pts, dst_pts):
        #Direct Linear Transform for homography estimation
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            xp, yp = dst_pts[i]
            #for each point correspondence, append two rows to matrix A
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)#SVD on A to find the 
        #eigenvector corresponding to the smallest eigenvalue
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2] if H[2, 2] != 0 else None #noramlizing
 

    def apply_homography(self, image, homography, canvas_size):
        print("entered apply homo")
        h, w = canvas_size
        warped_image = np.zeros((h, w, 3), dtype=np.uint8)
        inv_homography = np.linalg.inv(homography)

        for y in range(h):
            for x in range(w):
                # Map canvas coordinates back to image coordinates
                src_coords = inv_homography @ np.array([x, y, 1])
                src_coords = src_coords / src_coords[2]  # Normalizing

                # Check if coordinates fall within the image bounds
                sx, sy = int(src_coords[0]), int(src_coords[1])
                if 0 <= sx < image.shape[1] and 0 <= sy < image.shape[0]:
                    warped_image[y, x] = image[sy, sx]

        return warped_image

    def blend_images(self, canvas, image, offset):
        print("entered blennd")
        h, w, _ = image.shape
        cx, cy = offset

        for y in range(h):
            for x in range(w):
                if np.any(image[y, x] > 0):  # Only blend non-black pixels
                    # Blend pixel by pixel using simple averaging
                    canvas[cy + y, cx + x] = (canvas[cy + y, cx + x] // 2 + image[y, x] // 2)
