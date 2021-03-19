#!/bin/bash

set -ex

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

repo_root=$(mktemp -d)
rsync -av "${DOC_ROOT}/" "${repo_root}/"

pushd "${repo_root}"

git init
git remote add deploy "https://token:${TOKEN}@github.com/${DOC_REPO}.git"

touch .nojekyll
echo "www.kymat.io" > CNAME

git add .

msg="Pushing the docs for commit ${GITHUB_SHA} made on from ${GITHUB_REF} by ${GITHUB_ACTOR}"

git commit -am "${msg}"

git push deploy master --force

popd

exit 0
