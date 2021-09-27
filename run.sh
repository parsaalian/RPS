for f in ./config/random/test/*.yaml; do
    echo "$f"
    python3 ./run.py -c "$f" -t
done