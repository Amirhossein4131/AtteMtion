
# Create tmp folder if it doesn't exist
mkdir -p tmp

# Copy files to tmp folder and rename them
for i in {1..100}; do
  cp "${i}.example" "tmp/$(($i + 11000)).example"
done

