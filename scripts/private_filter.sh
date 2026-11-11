#!/usr/bin/env bash
# Git clean/smudge filter that keeps site-specific values (user name, host names,
# Slurm account) out of the repository. The working tree holds the real values so
# the scripts run as they are; commits hold placeholders.
#
# The mapping is a local, untracked file, .git/private-map: one "real<TAB>placeholder"
# pair per line, longest real value first. Without that file the filter passes
# content through unchanged, so a fresh clone simply shows the placeholders.
#
# One-time setup in a working copy that has the map:
#   git config filter.private.clean  'scripts/private_filter.sh clean'
#   git config filter.private.smudge 'scripts/private_filter.sh smudge'
#   git config filter.private.required true
set -euo pipefail
mode=${1:?clean or smudge}
map="$(git rev-parse --git-dir)/private-map"
[ -f "$map" ] || exec cat
MODE="$mode" MAP="$map" exec perl -e '
  my @pairs;
  open(my $fh, "<", $ENV{MAP}) or die "private-map: $!";
  while (my $line = <$fh>) {
    chomp $line;
    next if $line =~ /^\s*(#|$)/;
    my ($real, $placeholder) = split /\t/, $line, 2;
    die "bad line in private-map: $line\n" unless defined $placeholder && length $real && length $placeholder;
    push @pairs, $ENV{MODE} eq "clean" ? [$real, $placeholder] : [$placeholder, $real];
  }
  local $/;
  my $text = <STDIN>;
  for my $p (@pairs) {
    my ($from, $to) = @$p;
    # whole tokens only: not inside a longer run of letters and digits
    $text =~ s/(?<![A-Za-z0-9])\Q$from\E(?![A-Za-z0-9])/$to/g;
  }
  print $text;
'
