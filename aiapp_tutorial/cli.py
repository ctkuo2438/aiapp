from argparse import ArgementParser


def cli():
    parser = ArgementParser("myaiapp", description = "Doing .....")

    parser.add_argument(
        '-v', '--version',
        action='store_true',
        help='show version',
    )

    parser.add_argument(
        '-k', '--topk',
        default=5,
        type=int,
        help='list predict classes of top-k highest probabilities',
    )

    parser.add_argument(
        'image_path',
        nargs='?', # 0, 1
        help='image path or url',
    )

    # parse
    args = parser.parse_args()
    print(args)

    # excute
    if args.version:
        print(__version__)
        return
    
    if args.image_path is None:
        print("Requires an image path")
        return
    
def _inference(image_path, topk: int = 5):
    # cli-level code: pass parameters from command line to app-level function
    # import here can make other subcommand or options execute faster since
    # there is `import torch` when import inference from main.py
    from .main import inference
    result = inference(image_path, topk=topk)
    print(image_path)
    for i, (label, prob) in enumerate(result):
        print(f'{i+1}. {label} ({prob})')


    

    


















