import argparse
import render as r

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--thread_count',
        type=int,
        nargs='?',
        default=8,
        help="No. of threads for multi-threaded rendering. (Default: 8)"
    )
    parser.add_argument(
        '--sample_count',
        type=int,
        nargs='?',
        default=1,
        help="No. of samples per pixel. (Default: 1)"
    )
    parser.add_argument(
        '--sample_type',
        type=int,
        nargs='?',
        default=8,
        help="Sample type, 1: importance sampling, 2: sampling importance resampling. (Default: 1)"
    )
    parser.add_argument(
        '--max_bounce',
        type=int,
        nargs='?',
        default=3,
        help="Max number of bounce. (Default: 3)"
    )
    parser.add_argument(
        "--sky", 
        type=str, 
        required=True,
        help="Environment map file name, i.e. snow_field_2_puresky_1k"
    )
    parser.add_argument(
        '--mlt',
        action='store_true'
    )
    args = parser.parse_args()
    r.render(args.thread_count, args.sky, args.sample_count, args.sample_type, args.mlt, args.max_bounce)