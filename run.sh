for f in ./config/random/*.yaml; do
    echo "$f"
    python3 ./run.py -c "$f"
done