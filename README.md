# Copernicus Covariate Reader

A Python workflow to retrieve Copernicus Marine environmental covariates for
spatial observations.

This project was developed to support species distribution modelling around the
Galician coast, NW Spain, using haul records collected by observers on board
fishing vessels. The same workflow can be adapted to other regions, observation
types, and Copernicus products.

## Purpose

Species distribution models often need environmental covariates matched to
biological observations: temperature, salinity, chlorophyll, oxygen,
productivity, or similar layers.

This project separates that task into a reproducible data-preparation workflow:

1. define the spatial domain;
2. identify the model grid cells that represent the observations;
3. download only the required Copernicus subsets;
4. convert the downloaded NetCDF files into tabular data;
5. preserve stable links between observations and environmental tiles.

The result is a self-contained environmental dataset that can be joined later to
haul, survey, tracking, or sampling data without repeatedly querying Copernicus.

## Scientific Use Case

The original use case was Galicia, NW Spain.

Input observations were fishing hauls, each with:

- a haul identifier;
- longitude and latitude;
- sampling date;
- fishing depth.

The workflow assigns each haul to the nearest valid sea grid cell from the
Copernicus product. Environmental data are then downloaded for the selected tile
and date, down to the deepest haul associated with that tile.

Although the current workflow is configured for daily data, the same design can
be adapted to monthly, seasonal, annual, or custom time windows.

## Main Concepts

### Region and Bounding Boxes

The user defines a broad region of interest using longitude and latitude limits.

For large areas, the region can be split into latitude bands. This is useful
because longitude degrees do not represent the same physical distance at all
latitudes. Splitting the region reduces distortion when spatial distances are
approximated.

The number of latitude bands is configurable. The repository also includes helper
logic to estimate a suitable number of bands from grid resolution and acceptable
spatial error.

### Sea and Land Cells

The workflow uses the static layer associated with a Copernicus product to
determine which grid cells are sea and which are land.

Only sea cells are assigned valid tile identifiers. Land cells are excluded from
the tile catalogue, so observations are matched to the nearest sea cell rather
than simply to the nearest grid coordinate.

This is important in coastal areas such as Galicia, where small spatial
differences can place a point close to land, estuaries, or complex coastlines.

### Nearest Tile Assignment

Each observation is assigned to the nearest sea tile using a KD-tree.

Longitude is scaled by the cosine of latitude before distance calculations. This
gives a local approximation to Euclidean distance while working from longitude
and latitude coordinates.

This approach is appropriate when observations are matched to nearby grid-cell
centres. It can also be used over larger domains if the search is structured so
that candidate tiles remain geographically close to the observations they
represent. If observations are sparse and potential matches may be far away,
curvature and geodesic distance become more important and should be considered
explicitly.

The output is a table linking each observation to:

- `tile_id`;
- tile centre longitude;
- tile centre latitude.

For the Galicia case, the observation identifier is `haul_id`, but the same idea
applies to `survey_id`, `station_id`, `sample_id`, or any other observation key.

### Depth Handling

For each tile, the workflow identifies the deepest observation assigned to that
tile.

Copernicus data are then requested from the surface down to that depth. This
avoids downloading unnecessary vertical layers while retaining the environmental
range needed to represent all observations associated with the tile.

### Time Handling

The workflow was designed for daily environmental covariates. Each tile-date
combination becomes a download request.

The current configuration uses daily requests because the target products and
modelling workflow were daily, but the same structure can be modified for other
temporal resolutions.

### Download Strategy

The downloader requests small Copernicus subsets around each selected tile centre
rather than downloading large regional cubes.

This makes the workflow scalable when only a subset of grid cells is needed. It
also keeps the environmental data independent from the later modelling steps:
once downloaded, the data can be reused, audited, joined, or transferred without
depending on the original observation table.

Concurrent downloads are supported through a configurable maximum concurrency
setting, which is useful because Copernicus allows a limited number of parallel
requests.

### NetCDF to CSV Conversion

Downloaded NetCDF files are converted into CSV tables.

The final CSV outputs keep the tile identifiers, time, depth, and environmental
variables together, making them easy to join back to biological observations.

## Workflow Overview

1. Download the static product layer.
2. Build the tile source data:
   - assign observations to sea tiles;
   - build the tile-date request table;
   - identify the deepest required depth per tile.
3. Fetch Copernicus data for each tile-date-depth request.
4. Convert NetCDF files to CSV.
5. Amalgamate CSV files into a final tabular dataset.
6. Optionally zip and upload the product output.

## Inputs

Typical inputs are:

- an observation table with IDs, coordinates, dates, and depths;
- a static Copernicus mask layer;
- Copernicus product configuration;
- region limits;
- spatial resolution;
- number of latitude bands;
- download/conversion options.

## Outputs

The main outputs are:

- observation-to-tile mapping;
- tile-date-depth request table;
- downloaded NetCDF files;
- per-tile CSV files;
- an amalgamated CSV table ready for downstream modelling;
- optional compressed archive for storage or transfer.

## Configuration

The workflow is configured through environment variables, including:

- input and output paths;
- Copernicus dataset ID;
- variable names;
- spatial domain;
- spatial resolution;
- latitude band count;
- maximum download concurrency;
- product and file naming options.

## Repository Entry Points

The main scripts are:

- `src/actions/download_static_layer.py`
- `src/actions/build_tiles_source_data.py`
- `src/actions/fetch_copernicus_data.py`
- `src/actions/zip_and_upload_to_s3.py`

## Notes and Limitations

- The current implementation assumes rectilinear Copernicus grids with
  one-dimensional longitude and latitude coordinates.
- Land/sea classification depends on the Copernicus static mask for the selected
  product.
- Distance calculations use longitude scaling by latitude and are intended for
  cases where matched grid cells are geographically close to the observations.
- For sparse observations or large candidate-search distances, geodesic distance
  should be considered.
- The Galicia workflow uses haul IDs, but the same design applies to other
  observation identifiers.
- The default workflow is daily, but the same structure can be adapted to other
  temporal resolutions.
