import cupy as cp
import numpy as np
import cv2
import time
import random

def animate_squares():
    # Set up black screen
    height, width = 480, 640
    black_screen = np.zeros((height, width, 3), dtype=np.uint8)

    # Set up initial positions and velocities for squares
    square1_size = random.randint(20, 100)
    square2_size = random.randint(20, 100)

    square1_pos = [random.randint(0, width - square1_size), random.randint(0, height - square1_size)]
    square2_pos = [random.randint(0, width - square2_size), random.randint(0, height - square2_size)]

    square1_velocity = [random.randint(2, 5), random.randint(2, 5)]
    square2_velocity = [random.randint(2, 5), random.randint(2, 5)]

    # Animation loop
    while True:
        # Update square positions
        square1_pos[0] += square1_velocity[0]
        square1_pos[1] += square1_velocity[1]

        square2_pos[0] += square2_velocity[0]
        square2_pos[1] += square2_velocity[1]

        # Check boundary conditions for square1
        if square1_pos[0] < 0 or square1_pos[0] + square1_size > width:
            square1_velocity[0] = -square1_velocity[0]

        if square1_pos[1] < 0 or square1_pos[1] + square1_size > height:
            square1_velocity[1] = -square1_velocity[1]

        # Check boundary conditions for square2
        if square2_pos[0] < 0 or square2_pos[0] + square2_size > width:
            square2_velocity[0] = -square2_velocity[0]

        if square2_pos[1] < 0 or square2_pos[1] + square2_size > height:
            square2_velocity[1] = -square2_velocity[1]

        # Draw squares on the black screen
        black_screen = np.zeros((height, width, 3), dtype=np.uint8)  # Reset the screen
        cv2.rectangle(black_screen, tuple(square1_pos), (square1_pos[0] + square1_size, square1_pos[1] + square1_size),
                      (255, 255, 255), -1)  # Draw square1

        cv2.rectangle(black_screen, tuple(square2_pos), (square2_pos[0] + square2_size, square2_pos[1] + square2_size),
                      (255, 255, 255), -1)  # Draw square2

        cv2.imshow('Animating Squares', black_screen)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    animate_squares()


"""
def main():

    try:
        source_surface = ghetto_nvds.NvBufSurface(map_info)

        dest_array = cp.zeros(
            (source_surface.surfaceList[0].height, source_surface.surfaceList[0].width, 4),
            dtype=cp.uint8
        )

        dest_surface = ghetto_nvds.NvBufSurface(map_info)
        dest_surface.struct_copy_from(source_surface)
        assert(source_surface.numFilled == 1)
        assert(source_surface.surfaceList[0].colorFormat == 19) # RGBA

        dest_surface.surfaceList[0].dataPtr = dest_array.data.ptr

        dest_surface.mem_copy_from(source_surface)
        print(type(dest_array[:,:,0]))
        # print(type(dest_surface.surfaceList[0].dataPtr ))

        boxes = cp.zeros(
            (source_surface.surfaceList[0].height* source_surface.surfaceList[0].width , 4),
            dtype=cp.uint16
        )
        cudaCCLWrapper.PyprocessCCL(dest_array[:,:,0],width,height,boxes)
        print(cp.asnumpy(boxes))

        # print(boxes)
    finally:
        buf.unmap(map_info)"""