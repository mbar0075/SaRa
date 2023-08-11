import cv2
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from skimage import data
import plotly.graph_objects as go

def create_visual_plot_dashboard(input_image, grid_size, masks, mask_segments_min, heatmap):
    fixed_circle_size = 77
    img = input_image
    fig = make_subplots(2, 2)
    # Utilising go.Image since subplots require traces, whereas px functions return a figure
    fig.add_trace(go.Image(z=img), 1, 1)

    # Creating histogram
    for channel, color in enumerate(['red', 'green', 'blue']):
        fig.add_trace(go.Histogram(x=img[..., channel].ravel(), opacity=0.5,
                                marker_color=color, name='%s channel' % color, histnorm='percent'), row=1, col=2)

    combined_mask_image = np.zeros_like(img)

    for rank in mask_segments_min:
        mask = masks[rank]['mask']
        # Getting center of white area
        y, x = np.where(mask > 0)
        y_mean = int(np.mean(y))
        x_mean = int(np.mean(x))
        
        # Adding chape annotations to first subplot trace
        fig.add_shape(
            type="circle",
            x0=x_mean - fixed_circle_size / 2,
            y0=y_mean - fixed_circle_size / 2,
            x1=x_mean + fixed_circle_size / 2,
            y1=y_mean + fixed_circle_size / 2,
            line=dict(color='white', width=1),
            opacity=0.5,
            fillcolor='black'
        )

        fig.add_annotation(
            x=x_mean,
            y=y_mean,
            text=str(rank + 1),
            showarrow=False,
            font=dict(color='white', size=25),
            opacity=0.5,
        )

        # Creating a new image with the random color applied to white parts of the mask
        masked_image = combined_mask_image.copy()
        random_color = np.random.randint(0, 256, size=(1, 3), dtype=np.uint8)
        masked_image[mask > 0] = random_color

        # Combining the masked image with the original image
        combined_mask_image = cv2.bitwise_or(combined_mask_image, masked_image)

    fig.add_trace(go.Image(z=combined_mask_image), 2, 1)
    # heatmap, _ = sara.return_sara(cv2.cvtColor(img.copy(),cv2.COLOR_RGB2BGR), grid_size)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    fig.add_trace(go.Image(z=heatmap), 2, 2)

    # Updating subplot titles
    fig.update_xaxes(title_text="Original Image with Ranks", row=1, col=1)
    fig.update_xaxes(title_text="Image Histogram", row=1, col=2)
    fig.update_xaxes(title_text="Mask RCNN Masks", row=2, col=1)
    fig.update_xaxes(title_text="SaRa Heatmap", row=2, col=2)

    # Setting a title for the entire figure
    fig.update_layout(title="SaRa Visualiser Dashboard", title_x=0.5)
    # Setting figure height
    fig.update_layout(height=700)
    # fig.show()

    fig.write_html("output/plotly_dashboard.html")
