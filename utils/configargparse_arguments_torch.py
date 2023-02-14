import configargparse


def build_configargparser(parser):

    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--project_name", default=None, type=str)

    parser.add_argument("--base_folder_data_path", default="", required=False, type=str)
    parser.add_argument("--labelmap_path", default="", required=False, type=str)
    parser.add_argument("--data_dir_real_us", default="", required=False, type=str)
    parser.add_argument("--output_path", type=str, default="logs")
    parser.add_argument("--device", type=str, required=False, default="cuda")

    # Model
    parser.add_argument("--module", type=str, required=False)
    parser.add_argument("--outer_model", type=str, required=False)
    parser.add_argument("--inner_model", type=str, required=False)

        #Datatset
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--datamodule", type=str, required=False)
    parser.add_argument("--dataloader", type=str, required=False)
    parser.add_argument("--dataloader_real_us", type=str, required=False)
    parser.add_argument("--n_classes", type=int, required=False)
    parser.add_argument("--pred_label", type=int, required=False)


    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--global_learning_rate", type=float, required=False)

    parser.add_argument("--discr_model_learning_rate", type=float, required=False)
    parser.add_argument("--inner_model_learning_rate", type=float, required=False)
    parser.add_argument("--outer_model_learning_rate", type=float, required=False)
    parser.add_argument("--lambda_G_loss", type=float, required=False)


    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--min_epochs", default=10, type=int)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--validate_every_n_steps", default=1, type=int)
    

    parser.add_argument("--on_polyaxon", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--polyaxon_folder", default=None, type=str)
    parser.add_argument("--polyaxon_folder_real_us", default=None, type=str)



    # known_args, _ = parser.parse_known_args()
    known_args = parser.parse_args()
    return parser, known_args
