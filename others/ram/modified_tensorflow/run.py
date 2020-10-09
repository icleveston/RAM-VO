import argparse
from dataset import MNIST 
from ram import RAM

class mnist_config():
    name = 'mnist'
    output_path = 'out/'
    n_step = 6
    sample = 1
    glimpse_num = 8
    glimpse_scale = 1
    batch = 128
    epoch = 1000
    loc_std = 0.03
    unit_pixel = 12
    im_size = 28
    lr = 1e-3
    im_channel=1
    n_class=10
    max_grad_norm=5.0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', help='Run prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--dataset', type=str, default='mnist', help='Use the mnist')
    parser.add_argument('--step', type=int, default=1, help='Number of glimpse')
    parser.add_argument('--sample', type=int, default=1, help='Number of location samples during training')
    parser.add_argument('--glimpse', type=int, default=12, help='Glimpse base size')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='Max number of epoch')
    parser.add_argument('--load', type=int, default=100, help='Load pretrained parameters with id')
    parser.add_argument('--std', type=float, default=0.11, help='std of location')
    parser.add_argument('--pixel', type=int, default=26, help='unit_pixel')
    parser.add_argument('--scale', type=int, default=3, help='scale of glimpse')
    
    return parser.parse_args()

def main():
    
    # Read the input args
    FLAGS = get_args()

    if FLAGS.dataset == 'mnist':
  
        config = mnist_config()

        train_data = MNIST('train', batch_size=config.batch, shuffle=True, batch_dict_name=['data', 'label'])
        
        valid_data = MNIST('val', batch_size=config.batch, shuffle=True, batch_dict_name=['data', 'label'])
    
    elif FLAGS.dataset == 'regression_ball':
        pass

    # Create the model
    model = RAM(config)
    
    if FLAGS.train:
        
        model.create_train_model()

        # Train the model
        model.train(train_data, valid_data)
        
    elif FLAGS.predict:
        
        model.create_predict_model()

        model.predict(FLAGS.load)
    
    elif FLAGS.eval:
        
        model.create_predict_model()

        # Evaluate the model
        model.evaluate(FLAGS.load)
    

if __name__ == '__main__':
    
    main()