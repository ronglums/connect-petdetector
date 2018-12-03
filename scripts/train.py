
import argparse, os
import retrain as rt

parser = argparse.ArgumentParser()
parser.add_argument('--datastore-dir', type=str, dest='datastore_dir', help='datastore dir mounting point')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
args = parser.parse_args()

images_dir = os.path.join(args.datastore_dir, 'images')
bottleneck_dir = os.path.join(args.datastore_dir, 'bottleneck')
model_dir = os.path.join(args.datastore_dir, 'model')
output_dir = 'outputs'

rt.train(architecture='mobilenet_0.50_224', 
         image_dir=images_dir,
         output_dir=output_dir,
         bottleneck_dir=bottleneck_dir,
         model_dir=model_dir,
         learning_rate=args.learning_rate,
         use_hyperdrive=True)