for f in ./config/rps/*.yaml; do
    echo "$f"
    python3 ./run.py -c "$f" -t
done