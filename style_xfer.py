from absl import flags
from absl import app

# ------------------------------- flags -------------------------------

flags.DEFINE_enum(
    'method', 'wc', ['wc', 'hm'], 'Whitening-coloring or histogram matching'
)

FLAGS = flags.FLAGS


# ------------------------------- main --------------------------------


def main(argv):
    # Load encoder ----------------------------------------------------
    pass

    # Embed input volume ----------------------------------------------
    pass

    # Do feature transform --------------------------------------------
    pass

    # Load decoder ----------------------------------------------------
    pass

    # Decode transformed features -------------------------------------
    pass

    # Write to output volume ------------------------------------------
    pass


# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run(main)
