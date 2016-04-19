#!/bin/bash
echo "Running deployment script..."

cd ~
pip install pdoc

pdoc --html --html-dir ./doc carl
git clone -b master "https://${GH_TOKEN}@github.com/diana-hep/carl.github.io.git" deploy > /dev/null 2>&1 || exit 1

echo "Copying built files"
cp -r ./doc/* deploy

# Move into deployment directory
cd deploy

echo "Committing and pushing to GH"

git config user.name "Travis-CI"
git config user.email "travis@yoursite.com"

# Commit changes, allowing empty changes (when unchanged)
git add -A
git commit --allow-empty -m "Deploying site" || exit 1

# Push to branch
git push origin master > /dev/null 2>&1 || exit 1

echo "Pushed deployment successfully"
exit 0
