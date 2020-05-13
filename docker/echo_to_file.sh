#!/usr/bin/env bash
#check nr of arguments
[ "$#" -eq 2 ] || die "2 argument required, $# provided. Usage: echo_to_file.sh FILE LINE"

#echo into the file only if the line doesnt exist as per https://stackoverflow.com/a/28021305
FILE=$1
LINE=$2
grep -qF -- "$LINE" "$FILE" || echo "$LINE" >> "$FILE"
