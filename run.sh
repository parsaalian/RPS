for f in ./config/mantegna/*.yaml; do
    echo "$f"
    python3 ./run.py -c "$f"
done