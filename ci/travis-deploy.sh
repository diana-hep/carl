#!/bin/bash
echo "Running deployment script..."

# Generating documentation
cd ~
pip install pdoc==0.3.2
export XDG_CONFIG_HOME=${TRAVIS_BUILD_DIR}/ci/templates
pdoc --html --html-dir ./doc carl

# Copying to github pages
echo "Copying built files"
git clone -b gh-pages "https://${GH_TOKEN}@github.com/diana-hep/carl.git" deploy > /dev/null 2>&1 || exit 1
cp -r ./doc/carl/* deploy

# Move into deployment directory
cd deploy

# Commit changes, allowing empty changes (when unchanged)
echo "Committing and pushing to Github"
git config user.name "Travis-CI"
git config user.email "travis@yoursite.com"
git add -A
git commit --allow-empty -m "Deploying documentation for ${TRAVIS_COMMIT}" || exit 1

# Push to branch
git push origin gh-pages > /dev/null 2>&1 || exit 1

echo "Pushed deployment successfully"
exit 0
