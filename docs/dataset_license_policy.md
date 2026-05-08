# Dataset license & citation policy

TGraphX **does not redistribute third-party datasets**.

* Native synthetic datasets in `tgraphx.datasets.synthetic` are
  generated on-the-fly from a deterministic seed.  They are MIT-licensed
  along with the rest of TGraphX.
* Folder-backed datasets (`ImageFolderPatchGraphDataset`,
  `VolumeFolderPatchGraphDataset`) read **user-supplied files** from a
  user-supplied directory.  The license / provenance of those files is
  the user's responsibility.
* Optional adapters for torchvision / PyG / DGL / OGB delegate
  download and parsing to the upstream library; the upstream library's
  license terms apply.  TGraphX is only a converter.

## Cite the upstream dataset

Each adapter's `DatasetMetadata.citation` field points at the upstream
paper or dataset card.  When you publish results that depend on an
upstream dataset, cite that dataset (and TGraphX, if you wish) — see
`tgraphx.datasets.DatasetMetadata` for the exact citation string.

## What TGraphX never does

* No telemetry.
* No analytics.
* No cloud calls.
* No hidden downloads.
* No download at import time.
* No download during tests.
* No reading files outside the user-supplied root or cache root.
* No execution of remote pickle payloads.

## Network policy in tests

The TGraphX test suite never touches the network.  Any test that
exercises a download path uses `monkeypatch` to replace `urlopen` with
an in-memory fixture.

## Reporting issues

If a TGraphX adapter mishandles your data, license, or citation, open
an issue at <https://github.com/arashsajjadi/TGraphX/issues>.
