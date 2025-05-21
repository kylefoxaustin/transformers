import os
import time
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import autodistill components
from autodistill_owlv2 import OWLv2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot

def setup_environment():
    """Check and set up the environment for inference"""
    print("\n=== Environment Setup ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {properties.total_memory / 1e9:.2f} GB")
        print(f"CUDA Capability: {properties.major}.{properties.minor}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Create output directory for results
    os.makedirs("results", exist_ok=True)

    return device

def get_ontology():
    """Define the ontology (classes) for OWLv2"""
    ontology = CaptionOntology({
        "person": "person",
        "dog": "dog",
        "cat": "cat",
        "car": "car",
        "bicycle": "bicycle",
        "chair": "chair",
        "dining table": "dining_table",
        "bottle": "bottle",
        "laptop": "laptop",
        "cell phone": "cell_phone"
    })

    print(f"\n=== Model Ontology ===")
    print(f"Number of classes: {len(ontology.classes())}")
    print(f"Classes: {', '.join(ontology.classes())}")

    return ontology

def download_torchvision_dataset(num_images=25):
    """Download and prepare test images from torchvision datasets"""
    print(f"\n=== Downloading TorchVision Dataset ===")
    os.makedirs("test_images", exist_ok=True)

    # Choose the dataset (ImageNet is good as it has many classes)
    try:
        # First try to use ImageNet (requires manual download due to licensing)
        imagenet_dir = './data/imagenet'
        if os.path.exists(os.path.join(imagenet_dir, 'val')):
            print("Using ImageNet validation set")
            transform = transforms.Compose([transforms.Resize((256, 256))])
            dataset = datasets.ImageNet(root='./data', split='val', transform=transform)
            dataset_name = "imagenet"
        else:
            # Fall back to CIFAR-100 which is automatically downloaded
            print("ImageNet not found, using CIFAR-100 instead")
            transform = transforms.Compose([transforms.Resize((256, 256))])
            dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
            dataset_name = "cifar100"
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Falling back to CIFAR-10")
        transform = transforms.Compose([transforms.Resize((256, 256))])
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        dataset_name = "cifar10"

    # Save a subset of images
    saved_images = []

    for i in tqdm(range(min(num_images, len(dataset))), desc="Saving test images"):
        # Get image and label
        try:
            if dataset_name == "imagenet":
                img, label = dataset[i]
                class_name = dataset.classes[label].split(',')[0].replace(' ', '_')
            else:
                img, label = dataset[i]
                class_name = dataset.classes[label]

            # Convert and save image
            img_path = os.path.join("test_images", f"{dataset_name}_{class_name}_{i}.jpg")

            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL image if needed
                img = transforms.ToPILImage()(img)

            # Save the image
            img = img.convert('RGB')  # Ensure it's RGB
            img.save(img_path)
            saved_images.append(img_path)
        except Exception as e:
            print(f"Error processing image {i}: {e}")

    print(f"Saved {len(saved_images)} test images to the 'test_images' directory")
    return saved_images

def download_coco_dataset(num_images_per_class=3):
    """Download and prepare test images from COCO dataset"""
    try:
        from pycocotools.coco import COCO
        import requests
        from zipfile import ZipFile
        import io
    except ImportError:
        print("pycocotools not installed. Install with: pip install pycocotools")
        return []

    print(f"\n=== Downloading COCO Dataset ===")
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("coco_data", exist_ok=True)

    # Path to annotation file
    annotation_file = "coco_data/instances_val2017.json"

    # Download annotations if they don't exist
    if not os.path.exists(annotation_file):
        print("Downloading COCO annotations...")
        annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        try:
            response = requests.get(annotation_url)

            if response.status_code == 200:
                z = ZipFile(io.BytesIO(response.content))
                z.extractall("coco_data")
                print("Downloaded and extracted COCO annotations")
            else:
                print(f"Failed to download COCO annotations: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error downloading annotations: {e}")
            return []

    # Initialize COCO API
    try:
        coco = COCO(annotation_file)
    except Exception as e:
        print(f"Error initializing COCO API: {e}")
        return []

    # Get image IDs for the classes in our ontology
    class_names = ["person", "dog", "cat", "car", "bicycle", "chair",
                  "dining table", "bottle", "laptop", "cell phone"]
    image_ids = set()

    for class_name in class_names:
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            ids = coco.getImgIds(catIds=cat_ids)
            image_ids.update(ids[:num_images_per_class])  # Take up to n images per class

    # Download the images
    saved_images = []
    for img_id in tqdm(image_ids, desc="Downloading COCO images"):
        img_info = coco.loadImgs(img_id)[0]
        img_url = f"http://images.cocodataset.org/val2017/{img_info['file_name']}"
        img_path = os.path.join("test_images", f"coco_{img_info['file_name']}")

        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    saved_images.append(img_path)
                else:
                    print(f"Failed to download {img_url}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {img_url}: {e}")
        else:
            saved_images.append(img_path)

    print(f"Downloaded {len(saved_images)} COCO images to 'test_images' directory")
    return saved_images

def benchmark_single_image(model, image_path, warmup=True):
    """Benchmark inference time on a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image: {image_path}")
        return None, 0

    # Warmup (optional)
    if warmup:
        try:
            _ = model.predict(image_path)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Warmup failed for {image_path}: {e}")

    # Measure inference time
    try:
        start_time = time.time()
        results = model.predict(image_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - start_time

        return results, inference_time
    except Exception as e:
        print(f"Error during inference on {image_path}: {e}")
        return None, 0

def benchmark_batch(model, image_paths, num_runs=5):
    """Benchmark inference across multiple images and runs"""
    results_dict = {}
    fps_per_image = {}
    all_times = []
    successful_images = 0

    print(f"\n=== Benchmarking {len(image_paths)} images ({num_runs} runs each) ===")

    for image_path in tqdm(image_paths, desc="Processing images"):
        image_name = os.path.basename(image_path)

        # Check if image exists and can be opened
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Skipping {image_name} - unable to load image")
            continue

        times = []
        all_results = []

        for run in range(num_runs + 1):  # +1 for warmup
            is_warmup = (run == 0)
            results, inference_time = benchmark_single_image(model, image_path, warmup=False)

            if results is None:
                print(f"Warning: Skipping run {run} for {image_name} due to inference failure")
                continue

            if not is_warmup:  # Skip the warmup run in calculations
                times.append(inference_time)
                all_times.append(inference_time)
                all_results.append(results)

        if len(times) > 0:
            # Calculate statistics for this image
            mean_time = np.mean(times)
            mean_fps = 1.0 / mean_time
            fps_per_image[image_name] = mean_fps

            # Store best result (usually the last one is best due to caching)
            results_dict[image_path] = all_results[-1]
            successful_images += 1

    if successful_images == 0:
        print("No images were successfully processed")
        return 0, {}, {}

    # Calculate overall statistics
    overall_mean_time = np.mean(all_times)
    overall_mean_fps = 1.0 / overall_mean_time

    print(f"\n=== Benchmark Results ===")
    print(f"Successfully processed {successful_images}/{len(image_paths)} images")
    print(f"Overall average: {overall_mean_time:.4f}s ({overall_mean_fps:.2f} FPS)")

    # Show top 5 fastest and slowest images
    sorted_fps = sorted(fps_per_image.items(), key=lambda x: x[1])

    if len(sorted_fps) >= 2:
        print("\nSlowest images:")
        for img, fps in sorted_fps[:min(5, len(sorted_fps))]:
            print(f"  {img}: {fps:.2f} FPS")

        print("\nFastest images:")
        for img, fps in sorted_fps[-min(5, len(sorted_fps)):]:
            print(f"  {img}: {fps:.2f} FPS")

    return overall_mean_fps, fps_per_image, results_dict

def visualize_detections(model, image_path, results, output_dir="results"):
    """Visualize detection results and save the image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot visualize {image_path} - unable to load image")
            return None

        # Plot the results
        try:
            plot_img = plot(
                image=image,
                classes=model.ontology.classes(),
                detections=results
            )
        except Exception as e:
            print(f"Error plotting results for {image_path}: {e}")
            return None

        # Save the visualized result
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"detected_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, plot_img)

        # Return the path to the saved image
        return output_path
    except Exception as e:
        print(f"Error visualizing {image_path}: {e}")
        return None

def plot_benchmark_results(fps_per_image, output_dir="results"):
    """Create a visual plot of benchmark results"""
    try:
        if not fps_per_image:
            print("No benchmark data to plot")
            return None

        # Only plot up to 20 images to keep the chart readable
        if len(fps_per_image) > 20:
            # Get 20 evenly spaced samples
            items = list(fps_per_image.items())
            indices = np.linspace(0, len(items)-1, 20).astype(int)
            selected_items = [items[i] for i in indices]
            fps_per_image = dict(selected_items)

        plt.figure(figsize=(12, 6))
        names = list(fps_per_image.keys())
        fps_values = list(fps_per_image.values())

        # Create bars with different colors based on FPS
        colors = plt.cm.viridis(np.array(fps_values) / max(fps_values))
        bars = plt.bar(range(len(fps_values)), fps_values, color=colors)

        # Add average line
        avg_fps = np.mean(fps_values)
        plt.axhline(y=avg_fps, color='r', linestyle='--', label=f'Average: {avg_fps:.2f} FPS')

        plt.xlabel('Image')
        plt.ylabel('Frames Per Second (FPS)')
        plt.title('OWLv2 Inference Speed on Different Images')
        plt.xticks(range(len(names)), [name[:15] + '...' if len(name) > 15 else name for name in names], rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()

        # Save the plot
        output_path = os.path.join(output_dir, "owlv2_fps_benchmark.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved benchmark plot to {output_path}")

        # Close the figure to free memory
        plt.close()

        return output_path
    except Exception as e:
        print(f"Error creating benchmark plot: {e}")
        return None

def create_html_report(model_info, benchmark_results, visualization_paths, plot_path, output_dir="results"):
    """Create an HTML report with all results"""
    try:
        if not os.path.exists(plot_path):
            print(f"Warning: Plot file not found at {plot_path}")
            plot_basename = "Missing plot file"
        else:
            plot_basename = os.path.basename(plot_path)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>OWLv2 Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .detection-images {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .detection-card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; width: 300px; }}
                .detection-card img {{ width: 100%; height: auto; }}
                .detection-info {{ padding: 10px; }}
                .benchmark-plot {{ margin: 20px 0; text-align: center; }}
                .benchmark-plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>OWLv2 Transformer Benchmark Report</h1>

            <div class="container">
                <h2>Model Information</h2>
                <table>
                    <tr><th>Hardware</th><td>{model_info['hardware']}</td></tr>
                    <tr><th>Classes</th><td>{', '.join(model_info['classes'])}</td></tr>
                    <tr><th>Number of Images</th><td>{model_info['num_images']}</td></tr>
                    <tr><th>Average FPS</th><td>{model_info['average_fps']:.2f} (images per second)</td></tr>
                    <tr><th>Average Inference Time</th><td>{1000/model_info['average_fps']:.2f} ms per image</td></tr>
                </table>
            </div>

            <div class="container">
                <h2>Benchmark Results</h2>
                <div class="benchmark-plot">
                    <img src="{plot_basename}" alt="Benchmark Plot">
                </div>
            </div>

            <div class="container">
                <h2>Detection Results</h2>
                <p>Showing {len(visualization_paths)} example detections:</p>
                <div class="detection-images">
        """

        # Add detection images to the report
        for img_path, detections in visualization_paths.items():
            # Skip entries with missing visualization path
            if 'visualization_path' not in detections or not detections['visualization_path']:
                continue

            vis_path = detections['visualization_path']
            if not os.path.exists(vis_path):
                continue

            detection_filename = os.path.basename(vis_path)
            num_detections = len(detections['results'])

            detection_info = "<br>".join([
                f"{label}: {conf:.2f}" for label, conf, _ in detections['results'][:5]
            ])

            if len(detections['results']) > 5:
                detection_info += f"<br>... and {len(detections['results']) - 5} more"

            html_content += f"""
                    <div class="detection-card">
                        <img src="{detection_filename}" alt="Detection results">
                        <div class="detection-info">
                            <h3>{os.path.basename(img_path)}</h3>
                            <p>Found {num_detections} detections</p>
                            <p>{detection_info}</p>
                        </div>
                    </div>
            """

        html_content += """
                </div>
            </div>
        </body>
        </html>
        """

        # Write the HTML report
        report_path = os.path.join(output_dir, "owlv2_benchmark_report.html")
        with open(report_path, "w") as f:
            f.write(html_content)

        print(f"Created HTML report at {report_path}")
        return report_path

    except Exception as e:
        print(f"Error creating HTML report: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="OWLv2 Transformer Benchmark")
    parser.add_argument("--dataset", type=str, choices=["torchvision", "coco"], default="torchvision",
                        help="Dataset to use for benchmarking (default: torchvision)")
    parser.add_argument("--num-images", type=int, default=25,
                        help="Number of images to use for benchmarking (default: 25)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of inference runs per image (default: 3)")
    parser.add_argument("--visualize-max", type=int, default=10,
                        help="Maximum number of images to visualize (default: 10)")
    args = parser.parse_args()

    # Set up environment and get device
    device = setup_environment()

    # Get ontology
    ontology = get_ontology()

    # Load model - Important fix: removed device parameter since OWLv2 doesn't accept it
    print("\n=== Loading OWLv2 Model ===")
    try:
        model = OWLv2(
            ontology=ontology
            # device parameter removed
        )
        print("Model loaded successfully")

        # Try to check if the model is using CUDA
        if torch.cuda.is_available():
            try:
                # This is hacky but might work on some implementations
                if hasattr(model, 'model'):
                    print(f"Model device: {next(model.model.parameters()).device}")
                else:
                    print("Could not determine model device")
            except:
                print("Could not determine if model is using CUDA")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Download dataset
    try:
        if args.dataset == "torchvision":
            image_paths = download_torchvision_dataset(num_images=args.num_images)
        else:  # coco
            image_paths = download_coco_dataset(num_images_per_class=max(1, args.num_images // 10))

        if not image_paths:
            print("No images available for benchmarking. Exiting.")
            return
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Run benchmarks
    try:
        avg_fps, fps_per_image, results_dict = benchmark_batch(
            model, image_paths, num_runs=args.num_runs
        )
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return

    # Visualize a subset of the results
    print(f"\n=== Visualizing Detections ===")
    visualization_paths = {}

    # Limit the number of visualizations to avoid creating too many files
    visualize_count = min(args.visualize_max, len(results_dict))
    images_to_visualize = list(results_dict.keys())[:visualize_count]

    for image_path in tqdm(images_to_visualize, desc="Creating visualizations"):
        results = results_dict[image_path]

        # Print detection results
        print(f"\nFound {len(results)} detections in {os.path.basename(image_path)}")
        for detection in results[:5]:  # Show only first 5 detections
            label, confidence, bbox = detection
            print(f"  Detected {label} with confidence {confidence:.4f} at {bbox}")

        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more detections")

        # Visualize detections
        vis_path = visualize_detections(model, image_path, results)
        if vis_path:
            visualization_paths[image_path] = {
                'visualization_path': vis_path,
                'results': results
            }

    # Create benchmark plot
    try:
        plot_path = plot_benchmark_results(fps_per_image)
        if not plot_path:
            plot_path = "owlv2_fps_benchmark.png"  # Fallback if plotting fails
    except Exception as e:
        print(f"Error creating benchmark plot: {e}")
        plot_path = "owlv2_fps_benchmark.png"  # Fallback

    # Create HTML report
    try:
        model_info = {
            'hardware': f"{'NVIDIA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            'classes': ontology.classes(),
            'num_images': len(results_dict),
            'average_fps': avg_fps if avg_fps > 0 else 0.01  # Avoid division by zero
        }

        html_report = create_html_report(
            model_info,
            fps_per_image,
            visualization_paths,
            plot_path
        )
    except Exception as e:
        print(f"Error creating HTML report: {e}")
        html_report = None

    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Number of images: {len(results_dict)}")
    print(f"Hardware: {model_info['hardware']}")
    print(f"Model: OWLv2 transformer")
    print(f"Number of classes: {len(ontology.classes())}")
    print(f"Average performance: {avg_fps:.2f} inferences per second")
    if html_report:
        print(f"Report generated: {html_report}")

if __name__ == "__main__":
    main()
