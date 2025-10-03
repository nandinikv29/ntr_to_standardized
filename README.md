This script converts an NTR vendor workbook into the Intellyse standard 3-sheet tariff format: Regions, Tariffs, Surcharges
The code reads the vendor Excel file (NTR-Freight-Tariffs.xlsx). and detects the header rows and cleans the tariff tables.
It builds a consistent list of regions with stable IDs and processes Express, Economy, and Domestic tariffs in one run.

It also merges everything into one workbook with three sheets:
Regions: all unique origins/destinations, Tariffs: all price bands with metadata (service type, route, dates, etc.), Surcharges: created empty (extension if needed).


What I did is, the script reads the Excel workbook and converts it into Intellyse’s standard three-sheet format. The first thing it does is look for the relevant sheets: Express, Economy, and Domestic. It then processes all of them in one run. For each sheet, it cleans up the raw table by detecting where the headers start, dropping junk columns, and standardizing the weight break columns into the format Intellyse expects, like 0_up_to_30[kg].

It then extracts all the unique region tokens, such as ‘Switzerland’ or ‘Zone A’, and assigns them stable IDs, which are written into the Regions sheet. Once regions are defined, it builds tariff rows by combining each origin–destination pair with the correct price bands and metadata like start date, service type, and currency.

At the end, it merges one Regions sheet, one Tariffs sheet containing all the Express, Economy, and Domestic data, and an empty Surcharges sheet as a placeholder. I also did it like it does some basic validation, like checking that tariff routes reference existing region IDs. The design is modular so it’s easy to extend later, for example to handle zone mapping or surcharges.

I believe my code is able to process multiple freight tariff categories, like Express, Economy, and Domestic in a single run. Instead of writing separate scripts for each, I designed it to load, parse, and merge all relevant sheets automatically. This means regions are compiled into a consistent list with stable IDs that stay the same every time you run the script, making the data reliable and easy to track across updates.

Also the code infers weight bands and limits even if the sheet headers change slightly or are formatted differently. It doesn't rely on fixed cell addresses or assumptions, rather it scans the data to find relevant headers and processes them dynamically. This makes the script adaptable to various input formats.

I also built the script with a defensive mindset. It detects messy or merged headers, logs warnings instead of crashing, and gracefully skips over invalid or ambiguous data. This robustness ensures the script works reliably on real-world, unpredictable Excel inputs without requiring manual cleanup.

Finally, I think the code is easy to extend. Whether you want to add support for new service types or improve validation checks, you can adapt and extend it easily.

I added a separate test file because the converter script may has to deal with messy and unpredictable Excel files. These files might have merged headers, weird number formats, or slightly different labels each time. So without tests, a small code change could accidentally break something important without our knowledge. 

The test file checks that all the small helper functions always behave the way we expect.

Requirements: Python 3.10+
Packages: pandas, openpyxl, typer, pytest