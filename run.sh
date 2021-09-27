for f in ./config/mantegna/test/*.yaml; do
    echo "$f"
    python3 ./run.py -c "$f" -t
done