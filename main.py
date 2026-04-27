# main.py

from TinyRelDetector.tinyrel_detector import ParseArgs, TrainDemo


if __name__ == "__main__":
    args = ParseArgs()
    TrainDemo(
        steps=args.steps,
        batch=args.batch,
        imgHw=tuple(args.img),
        dModel=args.d_model,
        numRoles=args.roles,
        dataYaml=args.data,
        split=args.split,
    )