import argparse
import datetime
import os
import pickle
from utils.solver import Solver, SequenceSolver
from utils.transformer import TModel, SimpleNet, TSequenceModel
from utils.data_loader import get_loader, get_sequence_loader
from utils.human_priors import HumanPrior

    
        
        
def main(args, train=True):
    os.makedirs(args.model_path, exist_ok=True)

    model = TModel(patch_dim=7, num_patches=3, out_dim=3, 
                dim=64, depth=5, heads=10, mlp_dim=64, 
                pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.)


    #model = SimpleNet()
    train_loader, test_loader = get_loader(args)

    solver = Solver(args, model, train_loader, test_loader)
    if train:
        solver.train()
    else:
        solver.load_model()
        solver.test()

def sequence_main(args, train=True):
    os.makedirs(args.model_path, exist_ok=True)

    model = TSequenceModel(patch_dim=7, num_patches=3, 
                force_patch_dim=7, force_num_patches=10,
                out_dim=3, 
                dim=64, depth=5, heads=10, mlp_dim=64, 
                pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.)

    """

    model = TModel(patch_dim=7, num_patches=3, out_dim=3, 
                dim=64, depth=5, heads=10, mlp_dim=64, 
                pool = 'cls', dim_head = 64, dropout = 0.01, emb_dropout = 0.)
    """

    #model = SimpleNet()
    train_loader, test_loader = get_sequence_loader(args)

    solver = SequenceSolver(args, model, train_loader, test_loader)

    #solver = Solver(args, model, train_loader, test_loader)
    if train:
        solver.train()
    else:
        solver.load_model()
        solver.test()


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument('--data_path', type=str, default='data/expert_data.pkl')
    parser.add_argument('--model_path', type=str, default='weights/')

    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    """
    f = open('config/config.yaml')
    args = yaml.load(f)
    args['model_path'] = os.path.join(args.model_path)
    print(args)
    
    #main(args)
    #sequence_main(args)
    main(args, train=False)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))