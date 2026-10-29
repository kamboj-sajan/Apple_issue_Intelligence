def values(matches, entity_type):
    return {m.value for m in matches if m.entity_type == entity_type}


def test_devices_and_models(extractor):
    m = extractor.extract({"title": "iPhone 15 and MacBook Pro with M2 chip; also an old MacBook and Mac mini"})
    assert {"iphone", "iphone:15", "macbook-pro", "apple-silicon", "apple-silicon:2", "macbook", "mac-mini"} <= values(
        m, "device"
    )
    assert "mac-pro" not in values(m, "device")  # "MacBook Pro" must not count as a Mac Pro


def test_os_versions_named_and_numeric(extractor):
    m = extractor.extract({"body": "Upgraded from High Sierra to macOS 14.2 (Sonoma), iPhone on iOS 17.1, OS X 10.11"})
    assert values(m, "os") == {"macos:10.13", "macos:14", "ios:17", "macos:10.11"}
    assert "macos:10.12" not in values(m, "os")  # "Sierra" inside "High Sierra" is not Sierra


def test_error_codes(extractor):
    m = extractor.extract(
        {
            "code": "The operation can't be completed (error -36). Error code: 0xE8000015. "
            "NSCocoaErrorDomain errno=13 OSStatus -50"
        }
    )
    assert {"error:-36", "error:0xe8000015", "domain:nscocoaerrordomain", "errno:13", "osstatus:-50"} <= values(
        m, "error"
    )
    assert "hex:0xe8000015" not in values(m, "error")  # not double-counted


def test_tech_entities_and_field_provenance(extractor):
    m = extractor.extract({"title": "Time Machine fails", "tags": "wifi bluetooth"})
    assert {"time-machine", "wifi", "bluetooth"} <= values(m, "tech")
    by_value = {x.value: x for x in m}
    assert by_value["time-machine"].field == "title"
    assert by_value["wifi"].field == "tags"


def test_categories_primary_and_uncategorized(extractor):
    cats, primary = extractor.categorize(["battery"], "Battery drains fast after update")
    assert primary == "battery-power"
    assert {"battery-power", "update-install"} <= {c.value for c in cats}
    assert extractor.categorize([], "Which font does this logo use?") == ([], "uncategorized")
