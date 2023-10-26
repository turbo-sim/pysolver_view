import subprocess

def run_sphinx_apidoc(output_dir, src_dir):
    cmd = ["sphinx-apidoc", "-o", output_dir, src_dir, '-e']
    subprocess.check_call(cmd)
    print("Sphinx apidoc completed successfully.")

def run_sphinx_build(docs_dir=".", build_dir="_build", builder="html"):
    cmd = ["sphinx-build", "-b", builder, docs_dir, build_dir]
    subprocess.check_call(cmd)
    print(f"Sphinx build ({builder} format) completed successfully.")

if __name__ == "__main__":
    run_sphinx_apidoc(output_dir="source/api/", src_dir="../pysolver_view")
    run_sphinx_build(docs_dir=".", build_dir="_build", builder="html")

