#!/bin/bash
echo "Running deployment script..."

cd ~
pip install pdoc

export XDG_CONFIG_HOME=${TRAVIS_BUILD_DIR}/ci/templates
pdoc --html --html-dir ./doc carl
git clone -b gh-pages "https://${GH_TOKEN}@github.com/diana-hep/carl.git" deploy > /dev/null 2>&1 || exit 1

echo "Copying built files"
cp -r ./doc/carl/* deploy

# Move into deployment directory
cd deploy

echo "Committing and pushing to GH"

git config user.name "Travis-CI"
git config user.email "travis@yoursite.com"

# Commit changes, allowing empty changes (when unchanged)
git add -A
git commit --allow-empty -m "Deploying site" || exit 1

# Push to branch
git push origin gh-pages > /dev/null 2>&1 || exit 1

echo "Pushed deployment successfully"
exit 0
