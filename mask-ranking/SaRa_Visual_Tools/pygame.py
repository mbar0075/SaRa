import pygame
import sys
import cv2
import numpy as np

def create_interactive_plot(input_image, mask_segments_min, heatmap):
    try:
        pygame.init()

        WIDTH = 1280
        HEIGHT = 720

        # Set the dimensions of the window
        original = cv2.resize(input_image, (WIDTH, HEIGHT))

        original_heatmap = cv2.resize(heatmap, (WIDTH, HEIGHT))


        screen_width = WIDTH
        screen_height = HEIGHT

        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Saliency Ranking App")

        # main_image = pygame.image.load('./output/original.png').convert_alpha()
        main_image = pygame.surfarray.make_surface(original.swapaxes(0, 1))
        main_image_heatmap = pygame.surfarray.make_surface(original_heatmap.swapaxes(0, 1))

        mask_imgs = []
        mask_fill_imgs = []

        for i in range(3):
            mask_imgs.append(pygame.image.load('./output/mask' + str(i) + '.png'))
            mask_imgs[i] = pygame.transform.scale(mask_imgs[i], (screen_width, screen_height))

            mask_fill_imgs.append(pygame.image.load('./output/mask_fill' + str(i) + '.png'))
            mask_fill_imgs[i] = pygame.transform.scale(mask_fill_imgs[i], (screen_width, screen_height))


        hovered_mask = None

        font = pygame.font.Font(None, 24)  # Create a font for displaying text

        running = True

        show_heatmap = False
        show_masks = False

        legend_font = pygame.font.Font(None, 32)  # Font for the legend

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.display.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:  # Listen for the spacebar key press
                        show_heatmap = not show_heatmap  # Toggle the show_heatmap state
                    elif event.key == pygame.K_m:
                        show_masks = not show_masks
                    elif event.key == pygame.K_q:
                        running = False
                        pygame.display.quit()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        pygame.display.quit()
                    

            screen.fill((0, 0, 0))  # Fill the screen with black

            # Blit main image and masks onto the screen
            if show_heatmap:
                screen.blit(main_image_heatmap, (0, 0))
            else:
                screen.blit(main_image, (0, 0))

            if show_masks:
                for mask in mask_imgs:
                    screen.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

            legend_text_heatmap = legend_font.render("H: Toggle Heatmap", True, (255, 255, 255))
            legend_text_masks = legend_font.render("M: Toggle Masks", True, (255, 255, 255))

            legend_width = max(legend_text_heatmap.get_width(), legend_text_masks.get_width())
            legend_height = legend_text_heatmap.get_height() + legend_text_masks.get_height()

            legend_bg = pygame.Surface((legend_width + 20, legend_height + 10))
            legend_bg.set_alpha(128)  # Set the alpha value to make it transparent
            legend_bg.fill((0, 0, 0))
            screen.blit(legend_bg, (screen_width - legend_bg.get_width(), 10))

            screen.blit(legend_text_heatmap, (screen_width - legend_text_heatmap.get_width() - 10, 15))
            screen.blit(legend_text_masks, (screen_width - legend_text_masks.get_width() - 10, 15 + legend_text_heatmap.get_height()))


            # for mask in mask_imgs:
            #     screen.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

            # Inside the main loop
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # Check if the mouse is hovering over a mask
            for i in range(len(mask_fill_imgs)):
                # check if mouse is hovering over mask i's white points
                if mask_fill_imgs[i].get_at((mouse_x, mouse_y)) == (255, 255, 255, 255):
                    hovered_mask = i
                    break
                else:
                    hovered_mask = None

            # Display the hovered mask's index
            if hovered_mask is not None:
                # black background, white font, display object index and rank
                title_text = font.render("Object " + str(hovered_mask + 1), True, (255, 255, 255))
                
                info = mask_segments_min[hovered_mask]
                rank = info[0]
                iou = np.round(info[1], 2)
                entropy = np.round(info[2], 2)

                rank_text = font.render("Rank: " + str(rank), True, (255, 255, 255))
                iou_text = font.render("IoU: " + str(iou), True, (255, 255, 255))
                entropy_text = font.render("Entropy: " + str(entropy), True, (255, 255, 255))

                width = entropy_text.get_width()
                height = title_text.get_height() + rank_text.get_height() + iou_text.get_height() + entropy_text.get_height()

                bg = pygame.Surface((width, height))

                bg.fill((16, 16, 16))

                inc = 10
                
                # Blit the text onto the background
                screen.blit(bg, (mouse_x + inc, mouse_y - inc))
                screen.blit(title_text, (mouse_x + inc, mouse_y - inc))
                screen.blit(rank_text, (mouse_x + inc, mouse_y - inc + title_text.get_height()))
                screen.blit(iou_text, (mouse_x + inc, mouse_y - inc + title_text.get_height() + rank_text.get_height()))
                screen.blit(entropy_text, (mouse_x + inc, mouse_y - inc + title_text.get_height() + rank_text.get_height() + iou_text.get_height()))
            
            pygame.display.flip()  # Update the display

    except SystemExit:
        print("Exiting...")
        pygame.quit()
        sys.exit()
