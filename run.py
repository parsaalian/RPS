import os
import sys
import traceback
from pprint import pprint

from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config


def main():
    args = parse_arguments()
    config = get_config(args.config_file, is_test=args.test)
    
    # log info
    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint(config)
    print("<" * 80)
    
    # Run the experiment
    try:
        runner = eval(config.runner)(config)
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()