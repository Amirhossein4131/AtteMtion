# Create tmp folder if it doesn't exist
mkdir -p tmp

# Copy files to tmp folder and rename them
for i in {1..100}; do
  cp "${i}.bin" "tmp/$(($i + 1400)).bin"
done

