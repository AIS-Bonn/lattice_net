#!/bin/sh -x

# AUTO_ADDED_PACKAGES=`apt-mark showauto`
#
# apt-get remove --purge -y $AUTO_ADDED_PACKAGES

apt-get autoremove -y

# . /build/cleanup.sh
rm -rf /tmp/* /var/tmp/*

apt-get clean
rm -rf /var/lib/apt/lists/*
